from common import np, torch, tensor
from mano_base import HandModel


def normals(vertices, faces):
    triangles = vertices[:, faces]
    face_normals = torch.cross(triangles[:, :, 1]-triangles[:, :, 0], triangles[:, :, 2]-triangles[:, :, 0], dim=-1)
    result = torch.zeros_like(vertices)
    for column in range(3):
        result.index_add_(1, torch.as_tensor(faces[:, column], dtype=torch.long), face_normals)
    return result/result.norm(dim=-1, keepdim=True).clamp_min(1e-10)


def robust(square):
    return 9*(torch.sqrt(1+square/9)-1)


class MarkerHand(HandModel):
    def set_anatomical_hinges(self,beta):
        from mano_base import FINGERS
        self.set_shape_axes(beta)
        vertices=self.template+torch.einsum('k,vck->vc',tensor(beta),self.shape)
        joints=self.regressor@vertices
        dorsal=torch.linalg.cross(joints[1]-joints[0],joints[7]-joints[0])
        dorsal=dorsal/dorsal.norm().clamp_min(1e-10)
        if (dorsal*self.normals[2:4].mean(0)).sum()<0:
            dorsal=-dorsal
        self.anatomical_flex_axes={}
        for chain in FINGERS.values():
            for joint in chain[:3]:
                direction=self.axes[joint-1]
                axis=-torch.linalg.cross(direction,dorsal)
                self.anatomical_flex_axes[joint-1]=axis/axis.norm().clamp_min(1e-10)

    def flex_axis(self,joint,normal):
        if hasattr(self,'anatomical_flex_axes'):
            return self.anatomical_flex_axes[joint]
        axis=-torch.linalg.cross(self.axes[joint],normal)
        return axis/axis.norm().clamp_min(1e-10)

    def set_shape_axes(self,beta):
        from mano_base import FINGERS
        vertices=self.template+torch.einsum('k,vck->vc',tensor(beta),self.shape)
        joints=self.regressor@vertices
        direction=normals(vertices[None],self.faces)[0]
        self.normals=direction[self.ids]
        axes=torch.zeros(15,3)
        for chain in FINGERS.values():
            for ordinal,joint in enumerate(chain[:3]):
                child=joints[chain[ordinal+1]] if ordinal<2 else vertices[chain[-1]]
                axes[joint-1]=child-joints[joint]
        self.axes=axes/axes.norm(dim=-1,keepdim=True).clamp_min(1e-10)

    def set_attachment(self, profile):
        self.attachment_faces=profile.get('marker_attachment_faces')
        self.attachment_barycentric=profile.get('marker_attachment_barycentric')

    def hinges(self):
        from mano_base import FINGERS
        selected = []
        axes = []
        for finger, chain in list(FINGERS.items())[:4]:
            normal = self.normals[dict(index=7, middle=9, pinky=13, ring=11)[finger]]
            for joint in chain[1:3]:
                axes.append(self.flex_axis(joint-1,normal))
                selected.append(joint-1)
        return selected, torch.stack(axes)

    def predict(self, beta, pose, orient, wrist, ids, heights, wrist_offsets):
        from smplx.lbs import batch_rodrigues
        vertices, joints = self.forward(beta, pose, orient, wrist, vertices=True)
        surface,direction=self.surface(vertices,ids)
        prediction = surface+heights[None, :, None]*direction
        rotation=orient if orient.ndim==3 else batch_rodrigues(orient)
        prediction = torch.cat([torch.einsum('bij,mj->bmi', rotation, wrist_offsets)+wrist[:, None], prediction[:, 2:]], 1)
        return vertices, joints, prediction

    def surface(self,vertices,ids):
        normal=normals(vertices,self.faces)
        if getattr(self,'attachment_faces',None) is not None:
            faces=torch.as_tensor(self.attachment_faces,dtype=torch.long)
            weights=tensor(self.attachment_barycentric)
            surface=(vertices[:,faces]*weights[None,:,:,None]).sum(2)
            direction=(normal[:,faces]*weights[None,:,:,None]).sum(2)
            direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(1e-10)
        else:
            surface=vertices[:,np.maximum(ids,0)]
            direction=normal[:,np.maximum(ids,0)]
        return surface,direction
