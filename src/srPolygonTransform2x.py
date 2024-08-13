# ----------------------------------------------------------------------------
#  SRPolygonTransform (c) by Thomas Mueller
#
#    This experimental Blender add-on implements the polygon-rendering 
#    method for special-relativistic visualization. It transforms a mesh 
#    object to show what an observer moving with a velocity close to 
#    the speed of light would see. However, only the geometric distortion is
#    taken into account and neither the Doppler effect nor any illumination
#    or shadows will be handled correctly.
# 
#    See e.g. "A Survey of Visualization Methods for Special Relativity" 
#    by Daniel Weiskopf for a brief description:
#     http://drops.dagstuhl.de/opus/volltexte/2010/2711/pdf/20.pdf
#
#    A more detailed explanation can be found in the book
#    "Spezielle und allgemeine Relativitätstheorie - Grundlagen, Anwendungen 
#    in Astrophysik und Kosmologie sowie relativistische Visualisierung" 
#    by Boblest, Mueller, and Wunner; DOI: 10.1007/978-3-662-47767-0
#
#
#  SRPolygonTransform is licensed under a Creative Commons Attribution-
#  NonCommercial-ShareAlike 4.0 International License.
#
#  A copy of the license can be found at 
#   <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
# ----------------------------------------------------------------------------

bl_info = {
    "name" : "SR Polygon Transformation",
    "author" : "Thomas Mueller",
    "version" : (0, 0, 1),
    "blender" : (2, 79, 0),
    "description" : "Special-relativistic polygon transformation of mesh object.",
    "category" : "Object"
}

import bpy
from mathutils import Vector
import numpy as np
from numpy.linalg import inv
   

# ------------------------------------------------------------
#  Operator class
# ------------------------------------------------------------
class ObjectSRPolygonTransform(bpy.types.Operator):
    """
    Special-relativistic polygon transformation
    """
    bl_idname = "object.sr_polygon_transform"
    bl_label = "SR Polygon Transformation"
    bl_options = {'REGISTER', 'UNDO'}

    # observation time
    t_obs = bpy.props.FloatProperty(
        name = "tobs", 
        description = "Observation time",
        step = 1,
        default = 0.0)

    # scaled velocity beta_x = v_x / c, where c is the speed of light
    #beta_x = bpy.props.FloatProperty(
    #    name = "vx", 
    #    description = "Scaled velocity in x-direction",
    #    default = 0.5, 
    #    step = 1,
    #    min = -0.99, 
    #    max = 0.99)

    beta_xyz = bpy.props.FloatVectorProperty(
        name = "velocity",
        description = "Scaled velocity",
        default = (0.5,0.0,0.0),
        subtype = "DIRECTION",
        unit='LENGTH')


    def kdelta(self,a,b):
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
        Lambda[0,0] = gamma
        for i in range(1,4):
            Lambda[0,i] = -gamma * beta[i-1]
            Lambda[i,0] = -gamma * beta[i-1]
            for j in range(1,4):
                Lambda[i,j] = self.kdelta(i,j) + gamma**2 / (gamma + 1) * beta[i-1] * beta[j-1]
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
        obj = context.selected_objects[0]

        if obj.type != 'MESH':
            self.report({'INFO'}, "Selected object has to be a mesh-object!")
            return {'FINISHED'}

        # Scaled velocity
        #beta = np.array([self.beta_x,0,0])
        beta = np.array(self.beta_xyz)

        # Lorentz transformation matrix for beta
        L = self.lorentzMatrix(beta)

        # Inverse Lorentz transformation matrix
        invL = inv(L)

        # Observer is located at origin within its own reference frame
        obs = np.array([self.t_obs,0,0,0])

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
            obj1 = invL.dot(obj2) + 0*a2
            v.co = Vector(obj1[1:])


    def execute(self, context):
        self.evalLT(context)

        return {'FINISHED'}
    

def register():
    bpy.utils.register_class(ObjectSRPolygonTransform)
    bpy.types.Scene.t_obs = bpy.props.FloatProperty(
        name = "tobs", 
        description = "Observation time",
        default = 0.0)


def unregister():
    bpy.utils.unregister_class(ObjectSRPolygonTransform)


if __name__ == "__main__":
    register()
