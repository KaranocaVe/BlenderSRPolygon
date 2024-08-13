# ----------------------------------------------------------------------------
#  SRPolygonTransform (c) by Thomas Mueller And KaranocaVe
#
#    本实验性Blender插件实现了用于特殊相对论可视化的多边形渲染方法。它通过变换网格对象来展示一个以接近光速移动的观察者所看到的情况。然而，此插件仅考虑几何扭曲，不处理多普勒效应以及任何光照或阴影的正确表现。
#
#    可参考 Daniel Weiskopf 的文章《A Survey of Visualization Methods for Special Relativity》了解简要描述：
#     http://drops.dagstuhl.de/opus/volltexte/2010/2711/pdf/20.pdf
#
#    更详细的解释可在 Boblest, Mueller, 和 Wunner 合著的书籍
#    《Spezielle und allgemeine Relativit?tstheorie - Grundlagen, Anwendungen in Astrophysik und Kosmologie sowie relativistische Visualisierung》中找到；DOI: 10.1007/978-3-662-47767-0
#
#
#  SRPolygonTransform 依据 Creative Commons Attribution-
#  NonCommercial-ShareAlike 4.0 International License 进行授权。
#
#  授权证书副本可在此处找到
#   <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
# ----------------------------------------------------------------------------

bl_info = {
    "name": "狭义相对论多边形变换",
    "author": "Thomas Mueller"",""KaranocaVe",
    "version": (0, 0, 2),
    "blender": (4, 0, 0),
    "description": "网格对象的狭义相对论多边形变换。",
    "category": "Object"
}

import numpy as np
from numpy.linalg import inv

import bpy
from mathutils import Vector


def lorentzMatrix(beta):
    """
    计算洛伦兹变换矩阵依赖于参数beta

    参数：
        beta：归一化速度 beta = v/c，其中 v 是物体速度，c 是光速，beta 是一个长度为3的数组或列表，表示速度分量。

    返回值:
        返回一个4x4的洛伦兹变换矩阵
    """
    gamma = 1 / np.sqrt(1 - np.dot(beta, beta))
    Lambda = np.identity(4)
    Lambda[0, 0] = gamma
    for i in range(1, 4):
        Lambda[0, i] = -gamma * beta[i - 1]
        Lambda[i, 0] = -gamma * beta[i - 1]
        for j in range(1, 4):
            Lambda[i, j] = (1 if i == j else 0) + (gamma ** 2 / (gamma + 1)) * beta[i - 1] * beta[j - 1]
    return Lambda


class ObjectSRPolygonTransform(bpy.types.Operator):
    """
        狭义相对论多边形变换
    """
    bl_idname = "object.sr_polygon_transform"
    bl_label = "狭义相对论多边形变换"

    def evalLT(self, context):
        scene = context.scene

        # 场景中的活动摄像机
        cam = scene.camera

        if len(context.selected_objects) == 0:
            self.report({'INFO'}, "未选择任何对象")
            return {'FINISHED'}

        for obj in context.selected_objects:
            if obj.type != 'MESH':
                self.report({'INFO'}, "请选择一个mesh对象")
                return {'FINISHED'}

            # 获取归一化速度
            beta = np.array(scene.beta_xyz)

            # 计算洛伦兹变换矩阵
            L = lorentzMatrix(beta)

            # 反转洛伦兹变换矩阵
            invL = inv(L)

            # 观察者位于其自身参考系的原点
            obs = np.array([scene.t_obs, 0, 0, 0])

            # 观察者的参考框架相对于全局框架有一个偏移
            a1 = np.append([0], cam.location)

            # 物体的参考框架相对于全局框架有一个偏移
            a2 = np.append([0], obj.location)

            # 将观察者变换到物体的静止参考系
            obs2 = L.dot(obs + a1 - a2)

            # 变换对象的每一个顶点
            for v in obj.data.vertices:
                # 将观察者的后向光锥与顶点的世界线相交
                dx = np.array(v.co) - obs2[1:]
                delta = np.sqrt(dx.dot(dx))

                # 光线需要从顶点发射的时间，以便在观察者的观测时间到达观察者
                tp2 = obs2[0] - delta

                # 创建包含顶点位置和发射时间的事件矢量
                arr = np.array([v.co[0], v.co[1], v.co[2]])
                obj2 = np.append([tp2], arr)

                # 使用逆洛伦兹变换矩阵将事件矢量转换到观察者的参考系
                obj1 = invL.dot(obj2)

                # 更新顶点坐标
                v.co = Vector(obj1[1:])

    def execute(self, context):
        self.evalLT(context)

        return {'FINISHED'}


class ObjectSRPolygonTransformPanel(bpy.types.Panel):
    """
    狭义相对论多边形变换面板
    """
    bl_label = "狭义相对论多边形变换面板"
    bl_idname = "OBJECT_PT_sr_polygon_transform"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tools'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene, 't_obs', text="观测时间")
        layout.prop(scene, 'beta_xyz', text="速度")
        layout.operator("object.sr_polygon_transform")


def register():
    bpy.utils.register_class(ObjectSRPolygonTransform)
    bpy.utils.register_class(ObjectSRPolygonTransformPanel)
    bpy.types.Scene.t_obs = bpy.props.FloatProperty(
        name="观测时间",
        min=0.0,
        default=0.0,
        unit="TIME_ABSOLUTE",
        description="观测时间",
        subtype="TIME",
        step=1
    )
    bpy.types.Scene.beta_xyz = bpy.props.FloatVectorProperty(
        name="速度/C",
        default=(0.5, 0.0, 0.0),
        size=3,
        min=-1.0,
        max=1.0,
        subtype="VELOCITY",
        unit="NONE",
        description="设置速度为光速的倍数"
    )


def unregister():
    bpy.utils.unregister_class(ObjectSRPolygonTransform)
    bpy.utils.unregister_class(ObjectSRPolygonTransformPanel)
    del bpy.types.Scene.t_obs
    del bpy.types.Scene.beta_xyz


if __name__ == "__main__":
    register()
