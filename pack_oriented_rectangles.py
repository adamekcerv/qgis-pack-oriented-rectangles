"""
***************************************************************************
*   Pack Oriented Rectangles into Polygons                                *
*   QGIS Processing Algorithm                                             *
*   -------------------------------------------------------------------   *
*   Solves the 2D polygon packing and nesting problem by determining      *
*   optimal rotation angles from the minimum bounding geometry,           *
*   testing orthogonal/diagonal orientations and grid phase offsets       *
*   to maximize inscribed rectangular module capacity.                    *
***************************************************************************
"""

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterDistance,
    QgsProcessingParameterFeatureSink,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsWkbTypes
)
import numpy as np
import shapely
from shapely import affinity

class PackOrientedRectanglesAlgorithm(QgsProcessingAlgorithm):
    INPUT = 'INPUT'
    WIDTH = 'WIDTH'
    LENGTH = 'LENGTH'
    OUTPUT_POLYGONS = 'OUTPUT_POLYGONS'
    OUTPUT_RECTANGLES = 'OUTPUT_RECTANGLES'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return PackOrientedRectanglesAlgorithm()

    def name(self):
        return 'packorientedrectangles'

    def displayName(self):
        return self.tr('Pack oriented rectangles into polygons')

    def group(self):
        return self.tr('Vector analysis')

    def groupId(self):
        return 'vectoranalysis'

    def shortHelpString(self):
        return self.tr(
            'Fits the maximum possible number of fixed-size rectangular modules '
            'into input polygon features using optimal 2D orientation and offset packing.\n\n'
            'The algorithm determines the optimal rotation angle from the polygon\'s '
            'minimum rotated bounding rectangle, tests orthogonal and diagonal orientations '
            'with grid phase shifts, and selects the configuration that maximizes interior capacity.\n\n'
            'Outputs:\n'
            '1. Evaluated polygons with capacity metrics (fits_module, module_count, packed_area_m2, coverage_pct, capacity_category).\n'
            '2. Individual packed rectangle polygon geometries fully contained within the parent polygons.'
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                self.tr('Input polygon layer'),
                [QgsProcessing.TypeVectorPolygon]
            )
        )
        self.addParameter(
            QgsProcessingParameterDistance(
                self.WIDTH,
                self.tr('Rectangle width [m]'),
                defaultValue=6.0,
                parentParameterName=self.INPUT
            )
        )
        self.addParameter(
            QgsProcessingParameterDistance(
                self.LENGTH,
                self.tr('Rectangle length [m]'),
                defaultValue=10.0,
                parentParameterName=self.INPUT
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_POLYGONS,
                self.tr('Evaluated polygons'),
                QgsProcessing.TypeVectorPolygon
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_RECTANGLES,
                self.tr('Packed rectangles'),
                QgsProcessing.TypeVectorPolygon
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        width = float(self.parameterAsDouble(parameters, self.WIDTH, context))
        length = float(self.parameterAsDouble(parameters, self.LENGTH, context))
        
        module_area = width * length

        # Output fields for evaluated polygons
        out_fields = QgsFields(source.fields())
        out_fields.append(QgsField('fits_module', QVariant.Int))
        out_fields.append(QgsField('module_count', QVariant.Int))
        out_fields.append(QgsField('packed_area_m2', QVariant.Double))
        out_fields.append(QgsField('coverage_pct', QVariant.Double))
        out_fields.append(QgsField('capacity_category', QVariant.String, len=100))

        (sink_polygons, dest_polygons_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT_POLYGONS,
            context,
            out_fields,
            source.wkbType(),
            source.sourceCrs()
        )

        # Output fields for individual packed rectangle modules
        rect_fields = QgsFields()
        rect_fields.append(QgsField('rect_id', QVariant.Int))
        rect_fields.append(QgsField('parent_id', QVariant.Int))
        rect_fields.append(QgsField('width_m', QVariant.Double))
        rect_fields.append(QgsField('length_m', QVariant.Double))
        rect_fields.append(QgsField('area_m2', QVariant.Double))

        (sink_rectangles, dest_rectangles_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT_RECTANGLES,
            context,
            rect_fields,
            QgsWkbTypes.Polygon,
            source.sourceCrs()
        )

        total_features = source.featureCount()
        rect_counter = 0

        for current, feat in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break

            geom = feat.geometry()
            if geom.isEmpty():
                continue

            # Convert QGIS geometry to Shapely via WKT
            poly = shapely.from_wkt(geom.asWkt())
            area = poly.area

            rects = []
            
            # 1. Total area feasibility filter
            if area >= module_area:
                # 2. Inscribed circle diameter filter (negative buffer erosion)
                core = poly.buffer(-width / 2.0)
                if not core.is_empty:
                    # 3. Minimum Rotated Rectangle principal angle
                    mrr = poly.minimum_rotated_rectangle
                    if mrr.geom_type == 'Polygon' and len(mrr.exterior.coords) >= 4:
                        c = mrr.exterior.coords
                        dx = c[1][0] - c[0][0]
                        dy = c[1][1] - c[0][1]
                        mrr_ang = float(np.degrees(np.arctan2(dy, dx)) % 180)
                    else:
                        mrr_ang = 0.0

                    if area > 20000.0:
                        test_angles = [mrr_ang]
                        shifts = [(0.0, 0.0)]
                    elif area > 5000.0:
                        test_angles = [mrr_ang]
                        shifts = [(0.0, 0.0), (width / 2.0, length / 2.0)]
                    else:
                        test_angles = list(set([0.0, mrr_ang, (mrr_ang + 45.0) % 180]))
                        shifts = [(0.0, 0.0), (width / 2.0, length / 2.0)]

                    minx, miny, maxx, maxy = poly.bounds
                    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
                    best_rects = []

                    # Test candidate rotations and phase shifts to maximize packed count
                    for ang in test_angles:
                        p_rot = affinity.rotate(poly, -ang, origin=(cx, cy)) if ang != 0.0 else poly
                        bx1, by1, bx2, by2 = p_rot.bounds

                        for w, l in [(width, length), (length, width)]:
                            for sx, sy in shifts:
                                xs = np.arange(bx1 + sx, bx2, w)
                                ys = np.arange(by1 + sy, by2, l)
                                if len(xs) == 0 or len(ys) == 0:
                                    continue
                                xv, yv = np.meshgrid(xs, ys)
                                b_arr = shapely.box(xv.ravel(), yv.ravel(), (xv + w).ravel(), (yv + l).ravel())
                                contained = shapely.contains(p_rot, b_arr)
                                if not np.any(contained):
                                    continue
                                valid_b = b_arr[contained]
                                if len(valid_b) > len(best_rects):
                                    if ang != 0.0:
                                        best_rects = [affinity.rotate(b, ang, origin=(cx, cy)) for b in valid_b]
                                    else:
                                        best_rects = list(valid_b)

                    # Centroid/representative point fallback for tight parcels
                    if len(best_rects) == 0:
                        for pt in [core.centroid, core.representative_point()]:
                            for ang in [0.0, mrr_ang, (mrr_ang + 45.0) % 180, (mrr_ang + 90.0) % 180]:
                                for w, l in [(width, length), (length, width)]:
                                    b = shapely.box(pt.x - w/2.0, pt.y - l/2.0, pt.x + w/2.0, pt.y + l/2.0)
                                    if ang != 0.0:
                                        b = affinity.rotate(b, ang, origin=(pt.x, pt.y))
                                    if poly.contains(b):
                                        best_rects = [b]
                                        break
                                if len(best_rects) > 0:
                                    break
                            if len(best_rects) > 0:
                                break

                    rects = best_rects

            cnt = len(rects)
            fits = 1 if cnt > 0 else 0
            packed_m2 = cnt * module_area
            cov_pct = min(100.0, (packed_m2 / area) * 100.0) if area > 0 else 0.0

            # Classification
            if cnt == 0:
                if area >= module_area:
                    category = f"Narrow corridor (width < {width:.1f} m)"
                else:
                    category = f"Insufficient area (< {module_area:.1f} m2)"
            elif cnt == 1:
                category = f"Single module (1 rectangle {width:.0f}x{length:.0f} m)"
            elif 2 <= cnt <= 5:
                category = "Small capacity (2 - 5 rectangles)"
            elif 6 <= cnt <= 20:
                category = "Medium capacity (6 - 20 rectangles)"
            else:
                category = "Large capacity (> 20 rectangles)"

            # Write polygon feature
            out_feat = QgsFeature(out_fields)
            out_feat.setGeometry(geom)
            attrs = list(feat.attributes()) + [fits, cnt, round(packed_m2, 2), round(cov_pct, 2), category]
            out_feat.setAttributes(attrs)
            sink_polygons.addFeature(out_feat, QgsFeatureSink.FastInsert)

            # Write individual packed rectangle features
            for r in rects:
                rect_counter += 1
                m_feat = QgsFeature(rect_fields)
                m_feat.setGeometry(QgsGeometry.fromWkt(r.wkt))
                m_feat.setAttributes([rect_counter, feat.id(), width, length, module_area])
                sink_rectangles.addFeature(m_feat, QgsFeatureSink.FastInsert)

            if total_features > 0:
                feedback.setProgress(int(current / total_features * 100))

        return {
            self.OUTPUT_POLYGONS: dest_polygons_id,
            self.OUTPUT_RECTANGLES: dest_rectangles_id
        }

