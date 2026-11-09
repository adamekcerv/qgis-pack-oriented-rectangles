"""
***************************************************************************
*   Pack Oriented Rectangles into Polygons                                *
*   QGIS Processing Algorithm                                             *
*   -------------------------------------------------------------------   *
*   Solves the 2D polygon packing and nesting problem by determining      *
*   optimal rotation angles from the minimum bounding geometry,           *
*   testing orthogonal/diagonal orientations and grid phase offsets       *
*   with optional greedy residual packing for absolute maximum capacity.  *
***************************************************************************
"""

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterDistance,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsFeatureSink,
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
    MODE = 'MODE'
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
            'into polygon features using optimal 2D orientation and offset packing.\n\n'
            'Packing Modes:\n'
            '1. Structured urban grid (Uniform orientation): Aligns all modules to the polygon\'s '
            'principal street/boundary axis with uniform orientation (recommended for realistic urban planning).\n'
            '2. Maximum greedy fill (Mixed orientations & residual pockets): After placing the primary grid, '
            'iteratively analyzes remaining vacant pockets and residual wings to pack additional modules '
            'under independent rotations (maximizes mathematical capacity, ideal for solar yards or parking).\n\n'
            'Outputs:\n'
            '1. Evaluated polygons with capacity metrics (fits_module, module_count, grid_modules, greedy_modules, packed_area_m2, coverage_pct, capacity_category).\n'
            '2. Individual packed rectangle polygon geometries with placement classification (Grid vs Greedy).'
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
            QgsProcessingParameterEnum(
                self.MODE,
                self.tr('Packing mode'),
                options=[
                    self.tr('Structured urban grid (Uniform orientation)'),
                    self.tr('Maximum greedy fill (Mixed orientations & residual pockets)')
                ],
                defaultValue=0
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

    def _find_best_grid(self, p, width, length, angles, shifts):
        if p is None or p.is_empty:
            return []
        if not p.is_valid:
            p = shapely.make_valid(p)

        minx, miny, maxx, maxy = p.bounds
        cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        best_rects = []

        for ang in angles:
            p_rot = affinity.rotate(p, -ang, origin=(cx, cy)) if ang != 0.0 else p
            if not p_rot.is_valid:
                p_rot = shapely.make_valid(p_rot)

            bx1, by1, bx2, by2 = p_rot.bounds

            for w, l in [(width, length), (length, width)]:
                for sx, sy in shifts:
                    xs = np.arange(bx1 + sx, bx2, w)
                    ys = np.arange(by1 + sy, by2, l)
                    if len(xs) == 0 or len(ys) == 0:
                        continue
                    xv, yv = np.meshgrid(xs, ys)
                    b_arr = shapely.box(xv.ravel(), yv.ravel(), (xv + w).ravel(), (yv + l).ravel())
                    
                    try:
                        contained = shapely.contains(p_rot, b_arr)
                    except Exception:
                        p_rot_c = shapely.make_valid(p_rot.buffer(0))
                        try:
                            contained = shapely.contains(p_rot_c, b_arr)
                        except Exception:
                            continue

                    if not np.any(contained):
                        continue
                    valid_b = b_arr[contained]
                    if len(valid_b) > len(best_rects):
                        if ang != 0.0:
                            best_rects = [affinity.rotate(b, ang, origin=(cx, cy)) for b in valid_b]
                        else:
                            best_rects = list(valid_b)

        return best_rects

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        width = float(self.parameterAsDouble(parameters, self.WIDTH, context))
        length = float(self.parameterAsDouble(parameters, self.LENGTH, context))
        mode = self.parameterAsEnum(parameters, self.MODE, context)
        
        module_area = width * length

        # Output fields for evaluated polygons
        out_fields = QgsFields(source.fields())
        out_fields.append(QgsField('fits_module', QVariant.Int))
        out_fields.append(QgsField('module_count', QVariant.Int))
        out_fields.append(QgsField('grid_modules', QVariant.Int))
        out_fields.append(QgsField('greedy_modules', QVariant.Int))
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
        rect_fields.append(QgsField('placement', QVariant.String, len=30))

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

            # Convert QGIS geometry to Shapely via WKT and sanitize
            poly = shapely.force_2d(shapely.from_wkt(geom.asWkt()))
            if not poly.is_valid:
                poly = shapely.make_valid(poly)
            area = poly.area

            grid_rects = []
            greedy_rects = []
            
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

                    angles = [mrr_ang] if area > 5000.0 else list(set([0.0, mrr_ang, (mrr_ang + 45.0) % 180]))
                    shifts = [(0.0, 0.0)] if area > 20000.0 else [
                        (0.0, 0.0),
                        (width / 2.0, 0.0),
                        (0.0, length / 2.0),
                        (width / 2.0, length / 2.0)
                    ]

                    grid_rects = self._find_best_grid(poly, width, length, angles, shifts)

                    # Centroid/representative point fallback for tight parcels
                    if len(grid_rects) == 0:
                        for pt in [core.centroid, core.representative_point()]:
                            for ang in [0.0, mrr_ang, (mrr_ang + 45.0) % 180, (mrr_ang + 90.0) % 180]:
                                for w, l in [(width, length), (length, width)]:
                                    b = shapely.box(pt.x - w/2.0, pt.y - l/2.0, pt.x + w/2.0, pt.y + l/2.0)
                                    if ang != 0.0:
                                        b = affinity.rotate(b, ang, origin=(pt.x, pt.y))
                                    if poly.contains(b):
                                        grid_rects = [b]
                                        break
                                if len(grid_rects) > 0:
                                    break
                            if len(grid_rects) > 0:
                                break

                    # Mode 1: Maximum Greedy Fill in residual pockets
                    if mode == 1 and len(grid_rects) > 0:
                        all_placed = list(grid_rects)
                        try:
                            residual = poly.difference(shapely.unary_union(grid_rects))
                            if not residual.is_valid:
                                residual = shapely.make_valid(residual)
                        except Exception:
                            residual = shapely.make_valid(poly).difference(shapely.make_valid(shapely.unary_union(grid_rects)))

                        max_passes = 3 if area > 5000.0 else 5
                        for _ in range(max_passes):
                            if residual.is_empty or residual.area < module_area:
                                break
                            if not residual.is_valid:
                                residual = shapely.make_valid(residual)

                            if residual.geom_type == 'Polygon':
                                sub_geoms = [residual]
                            elif residual.geom_type == 'MultiPolygon':
                                sub_geoms = list(residual.geoms)
                            elif residual.geom_type == 'GeometryCollection':
                                sub_geoms = [g for g in residual.geoms if g.geom_type == 'Polygon']
                            else:
                                sub_geoms = []

                            placed_in_pass = []
                            for comp in sorted(sub_geoms, key=lambda g: g.area, reverse=True):
                                if comp.area < module_area:
                                    continue
                                if not comp.is_valid:
                                    comp = shapely.make_valid(comp)
                                    if comp.geom_type != 'Polygon':
                                        continue

                                c_core = comp.buffer(-width / 2.0)
                                if c_core.is_empty:
                                    continue

                                m_sub = comp.minimum_rotated_rectangle
                                if m_sub.geom_type == 'Polygon' and len(m_sub.exterior.coords) >= 4:
                                    sc = m_sub.exterior.coords
                                    sub_ang = float(np.degrees(np.arctan2(sc[1][1] - sc[0][1], sc[1][0] - sc[0][0])) % 180)
                                else:
                                    sub_ang = 0.0

                                test_angs = list(set([sub_ang, (sub_ang + 90.0) % 180, (sub_ang + 45.0) % 180, 0.0]))
                                test_shifts = [
                                    (0.0, 0.0),
                                    (width / 2.0, 0.0),
                                    (0.0, length / 2.0),
                                    (width / 2.0, length / 2.0)
                                ]

                                comp_rects = self._find_best_grid(comp, width, length, test_angs, test_shifts)
                                if len(comp_rects) == 0:
                                    for pt in [c_core.centroid, c_core.representative_point()]:
                                        for ang in [0.0, sub_ang, (sub_ang + 45.0) % 180, (sub_ang + 90.0) % 180]:
                                            for w, l in [(width, length), (length, width)]:
                                                b = shapely.box(pt.x - w/2.0, pt.y - l/2.0, pt.x + w/2.0, pt.y + l/2.0)
                                                if ang != 0.0:
                                                    b = affinity.rotate(b, ang, origin=(pt.x, pt.y))
                                                try:
                                                    if comp.contains(b):
                                                        comp_rects = [b]
                                                        break
                                                except Exception:
                                                    if shapely.make_valid(comp).contains(b):
                                                        comp_rects = [b]
                                                        break
                                            if len(comp_rects) > 0:
                                                break
                                        if len(comp_rects) > 0:
                                            break

                                if len(comp_rects) > 0:
                                    placed_in_pass.extend(comp_rects)

                            if len(placed_in_pass) == 0:
                                break
                            greedy_rects.extend(placed_in_pass)
                            all_placed.extend(placed_in_pass)
                            try:
                                residual = poly.difference(shapely.unary_union(all_placed))
                            except Exception:
                                residual = shapely.make_valid(poly).difference(shapely.make_valid(shapely.unary_union(all_placed)))

            grid_cnt = len(grid_rects)
            greedy_cnt = len(greedy_rects)
            total_cnt = grid_cnt + greedy_cnt
            fits = 1 if total_cnt > 0 else 0
            packed_m2 = total_cnt * module_area
            cov_pct = min(100.0, (packed_m2 / area) * 100.0) if area > 0 else 0.0

            # Classification
            if total_cnt == 0:
                if area >= module_area:
                    category = f"Narrow corridor (width < {width:.1f} m)"
                else:
                    category = f"Insufficient area (< {module_area:.1f} m2)"
            elif total_cnt == 1:
                category = f"Single module (1 rectangle {width:.0f}x{length:.0f} m)"
            elif 2 <= total_cnt <= 5:
                category = "Small capacity (2 - 5 rectangles)"
            elif 6 <= total_cnt <= 20:
                category = "Medium capacity (6 - 20 rectangles)"
            else:
                category = "Large capacity (> 20 rectangles)"

            # Write polygon feature
            out_feat = QgsFeature(out_fields)
            out_feat.setGeometry(geom)
            attrs = list(feat.attributes()) + [fits, total_cnt, grid_cnt, greedy_cnt, round(packed_m2, 2), round(cov_pct, 2), category]
            out_feat.setAttributes(attrs)
            sink_polygons.addFeature(out_feat, QgsFeatureSink.FastInsert)

            # Write individual packed rectangle features
            for r in grid_rects:
                rect_counter += 1
                m_feat = QgsFeature(rect_fields)
                m_feat.setGeometry(QgsGeometry.fromWkt(r.wkt))
                m_feat.setAttributes([rect_counter, feat.id(), width, length, module_area, 'Grid'])
                sink_rectangles.addFeature(m_feat, QgsFeatureSink.FastInsert)

            for r in greedy_rects:
                rect_counter += 1
                m_feat = QgsFeature(rect_fields)
                m_feat.setGeometry(QgsGeometry.fromWkt(r.wkt))
                m_feat.setAttributes([rect_counter, feat.id(), width, length, module_area, 'Greedy'])
                sink_rectangles.addFeature(m_feat, QgsFeatureSink.FastInsert)

            if total_features > 0:
                feedback.setProgress(int(current / total_features * 100))

        return {
            self.OUTPUT_POLYGONS: dest_polygons_id,
            self.OUTPUT_RECTANGLES: dest_rectangles_id
        }
