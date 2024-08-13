# ----------------------------------------------------------------------------
#  SRPolygonTransform (c) by Thomas Mueller And KaranocaVe
#
#    ��ʵ����Blender���ʵ����������������ۿ��ӻ��Ķ������Ⱦ��������ͨ���任���������չʾһ���Խӽ������ƶ��Ĺ۲����������������Ȼ�����˲�������Ǽ���Ť���������������ЧӦ�Լ��κι��ջ���Ӱ����ȷ���֡�
#
#    �ɲο� Daniel Weiskopf �����¡�A Survey of Visualization Methods for Special Relativity���˽��Ҫ������
#     http://drops.dagstuhl.de/opus/volltexte/2010/2711/pdf/20.pdf
#
#    ����ϸ�Ľ��Ϳ��� Boblest, Mueller, �� Wunner �������鼮
#    ��Spezielle und allgemeine Relativit?tstheorie - Grundlagen, Anwendungen in Astrophysik und Kosmologie sowie relativistische Visualisierung�����ҵ���DOI: 10.1007/978-3-662-47767-0
#
#
#  SRPolygonTransform ���� Creative Commons Attribution-
#  NonCommercial-ShareAlike 4.0 International License ������Ȩ��
#
#  ��Ȩ֤�鸱�����ڴ˴��ҵ�
#   <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
# ----------------------------------------------------------------------------

bl_info = {
    "name": "��������۶���α任",
    "author": "Thomas Mueller"",""KaranocaVe",
    "version": (0, 0, 2),
    "blender": (4, 0, 0),
    "description": "����������������۶���α任��",
    "category": "Object"
}

import numpy as np
from numpy.linalg import inv

import bpy
from mathutils import Vector


def lorentzMatrix(beta):
    """
    ���������ȱ任���������ڲ���beta

    ������
        beta����һ���ٶ� beta = v/c������ v �������ٶȣ�c �ǹ��٣�beta ��һ������Ϊ3��������б���ʾ�ٶȷ�����

    ����ֵ:
        ����һ��4x4�������ȱ任����
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
        ��������۶���α任
    """
    bl_idname = "object.sr_polygon_transform"
    bl_label = "��������۶���α任"

    def evalLT(self, context):
        scene = context.scene

        # �����еĻ�����
        cam = scene.camera

        if len(context.selected_objects) == 0:
            self.report({'INFO'}, "δѡ���κζ���")
            return {'FINISHED'}

        for obj in context.selected_objects:
            if obj.type != 'MESH':
                self.report({'INFO'}, "��ѡ��һ��mesh����")
                return {'FINISHED'}

            # ��ȡ��һ���ٶ�
            beta = np.array(scene.beta_xyz)

            # ���������ȱ任����
            L = lorentzMatrix(beta)

            # ��ת�����ȱ任����
            invL = inv(L)

            # �۲���λ��������ο�ϵ��ԭ��
            obs = np.array([scene.t_obs, 0, 0, 0])

            # �۲��ߵĲο���������ȫ�ֿ����һ��ƫ��
            a1 = np.append([0], cam.location)

            # ����Ĳο���������ȫ�ֿ����һ��ƫ��
            a2 = np.append([0], obj.location)

            # ���۲��߱任������ľ�ֹ�ο�ϵ
            obs2 = L.dot(obs + a1 - a2)

            # �任�����ÿһ������
            for v in obj.data.vertices:
                # ���۲��ߵĺ����׶�붥����������ཻ
                dx = np.array(v.co) - obs2[1:]
                delta = np.sqrt(dx.dot(dx))

                # ������Ҫ�Ӷ��㷢���ʱ�䣬�Ա��ڹ۲��ߵĹ۲�ʱ�䵽��۲���
                tp2 = obs2[0] - delta

                # ������������λ�úͷ���ʱ����¼�ʸ��
                arr = np.array([v.co[0], v.co[1], v.co[2]])
                obj2 = np.append([tp2], arr)

                # ʹ���������ȱ任�����¼�ʸ��ת�����۲��ߵĲο�ϵ
                obj1 = invL.dot(obj2)

                # ���¶�������
                v.co = Vector(obj1[1:])

    def execute(self, context):
        self.evalLT(context)

        return {'FINISHED'}


class ObjectSRPolygonTransformPanel(bpy.types.Panel):
    """
    ��������۶���α任���
    """
    bl_label = "��������۶���α任���"
    bl_idname = "OBJECT_PT_sr_polygon_transform"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tools'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene, 't_obs', text="�۲�ʱ��")
        layout.prop(scene, 'beta_xyz', text="�ٶ�")
        layout.operator("object.sr_polygon_transform")


def register():
    bpy.utils.register_class(ObjectSRPolygonTransform)
    bpy.utils.register_class(ObjectSRPolygonTransformPanel)
    bpy.types.Scene.t_obs = bpy.props.FloatProperty(
        name="�۲�ʱ��",
        min=0.0,
        default=0.0,
        unit="TIME_ABSOLUTE",
        description="�۲�ʱ��",
        subtype="TIME",
        step=1
    )
    bpy.types.Scene.beta_xyz = bpy.props.FloatVectorProperty(
        name="�ٶ�/C",
        default=(0.5, 0.0, 0.0),
        size=3,
        min=-1.0,
        max=1.0,
        subtype="VELOCITY",
        unit="NONE",
        description="�����ٶ�Ϊ���ٵı���"
    )


def unregister():
    bpy.utils.unregister_class(ObjectSRPolygonTransform)
    bpy.utils.unregister_class(ObjectSRPolygonTransformPanel)
    del bpy.types.Scene.t_obs
    del bpy.types.Scene.beta_xyz


if __name__ == "__main__":
    register()
