import bpy
from mathutils import Vector
import numpy as np
from numpy.linalg import inv


class ObjectSRPolygonTransform(bpy.types.Operator):
    """
        Special-relativistic polygon transformation
        """
    bl_idname = "object.sr_polygon_transform"
    bl_label = "SR Polygon Transformation Func"

    def kdelta(self, a, b):
        """Kronecker delta"""
        return 1 if a == b else 0

    def lorentzMatrix(self, beta):
        """
            Generate Lorentz transformation matrix depending on beta.

            Args:
                beta: Scaled velocity beta = v/c where c is the speed of light.

            Return:
                Lorentz-Transformation matrix
            """
        gamma = 1 / np.sqrt(1 - np.dot(beta, beta))
        Lambda = np.identity(4)
        Lambda[0, 0] = gamma
        for i in range(1, 4):
            Lambda[0, i] = -gamma * beta[i - 1]
            Lambda[i, 0] = -gamma * beta[i - 1]
            for j in range(1, 4):
                Lambda[i, j] = self.kdelta(i, j) + gamma ** 2 / (gamma + 1) * beta[i - 1] * beta[j - 1]
        return Lambda

    def evalLT(self, context):
        scene = context.scene

        # Active camera within the scene
        cam = scene.camera

        if len(context.selected_objects) == 0:
            self.report({'INFO'}, "One object has to be selected!")
            return {'FINISHED'}

        if len(context.selected_objects) > 1:
            self.report({'INFO'}, "Select only one object!")
            return {'FINISHED'}

        # Selected object
        objraw = context.selected_objects[0]

        obj = objraw.copy()

        bpy.context.collection.objects.link(obj)

        if obj.type != 'MESH':
            self.report({'INFO'}, "Selected object has to be a mesh-object!")
            return {'FINISHED'}

        # Scaled velocity
        #beta = np.array([self.beta_x,0,0])
        beta = np.array(scene.beta_xyz)

        # Lorentz transformation matrix for beta
        L = self.lorentzMatrix(beta)

        # Inverse Lorentz transformation matrix
        invL = inv(L)

        # Observer is located at origin within its own reference frame
        obs = np.array([scene.t_obs, 0, 0, 0])

        # The observer's reference frame has an offset to the global frame
        a1 = np.append([0], cam.location)

        # The object's reference frame has an offset to the global frame
        a2 = np.append([0], obj.location)

        # Transform observer into the object's rest frame
        obs2 = L.dot(obs + a1 - a2)

        # Transform every vertex of the object
        for v in obj.data.vertices:
            # intersect observer's backward light cone with the world line
            # of the vertex
            dx = np.array(v.co) - obs2[1:]
            delta = np.sqrt(dx.dot(dx))

            # time when light has to be emitted from the vertex point in
            # order to reach the observer at his/her observation time
            tp2 = obs2[0] - delta

            # emission event
            obj2 = np.append([tp2], v.co)

            # transform emission event into the observer's reference frame
            # here, only the transformation into the global frame is necessary
            obj1 = invL.dot(obj2) + 0 * a2
            v.co = Vector(obj1[1:])

    def execute(self, context):
        self.evalLT(context)

        return {'FINISHED'}


class ObjectSRPolygonTransformPanel(bpy.types.Panel):
    """
    Special-relativistic polygon transformation panel
    """
    bl_label = "SR Polygon Transform Panel"
    bl_idname = "OBJECT_PT_sr_polygon_transform"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene, 't_obs', text="Observation Time")
        layout.prop(scene, 'beta_xyz', text="Velocity")
        layout.operator("object.sr_polygon_transform")


def updatefunc(self, context):
    ObjectSRPolygonTransform.execute(self, context)


def register():
    bpy.utils.register_class(ObjectSRPolygonTransform)
    bpy.utils.register_class(ObjectSRPolygonTransformPanel)
    bpy.types.Scene.t_obs = bpy.props.FloatProperty(
        name="Observation Time",
        default=0.0,
        step=1,
        update=updatefunc
    )
    bpy.types.Scene.beta_xyz = bpy.props.FloatVectorProperty(
        name="Velocity",
        default=(0.5, 0.0, 0.0),
        subtype="DIRECTION",
        unit='LENGTH',
        update=updatefunc
    )


def unregister():
    bpy.utils.unregister_class(ObjectSRPolygonTransform)
    bpy.utils.unregister_class(ObjectSRPolygonTransformPanel)
    del bpy.types.Scene.t_obs
    del bpy.types.Scene.beta_xyz


if __name__ == "__main__":
    register()
