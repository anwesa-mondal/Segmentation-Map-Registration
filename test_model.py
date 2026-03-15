"""
Smoke-tests for AffineNet, UNet, SpatialTransformer, and the full pipeline.
Uses small volumes (32^3) so it runs in seconds on CPU.

Run:  python test_model.py
"""

import sys
import os
import torch
import importlib.util

# All source files have spaces/parens in names, so use importlib throughout
_dir = os.path.dirname(os.path.abspath(__file__))

def _load(mod_name, filename):
    spec = importlib.util.spec_from_file_location(
        mod_name, os.path.join(_dir, filename)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod          # register so cross-imports work
    spec.loader.exec_module(mod)
    return mod

losses_2 = _load("losses_2", "losses_2 (1).py")
model_1 = _load("model_1", "model_1 (1).py")
cl = _load("compoundlossfunction_2", "compoundlossfunction_2 (1).py")

_angles_to_rotation_matrix = model_1._angles_to_rotation_matrix
params_to_affine = model_1.params_to_affine
affine_to_dense_displacement = model_1.affine_to_dense_displacement
AffineNet = model_1.AffineNet
UNet = model_1.UNet
SpatialTransformer = model_1.SpatialTransformer
compound_loss = cl.compound_loss

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


# -----------------------------------------------------------------------
# 1. Rotation matrix validity
# -----------------------------------------------------------------------
print("\n=== 1. Rotation matrix properties ===")

angles = torch.tensor([0.1, -0.2, 0.15])
R = _angles_to_rotation_matrix(angles)

check("R is 3x3", R.shape == (3, 3))
check("R is orthogonal (R^T R = I)",
      torch.allclose(R.T @ R, torch.eye(3), atol=1e-5),
      f"R^T R = {R.T @ R}")
check("det(R) = +1 (proper rotation)",
      torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-5),
      f"det = {torch.det(R).item():.6f}")

angles_zero = torch.zeros(3)
R_zero = _angles_to_rotation_matrix(angles_zero)
check("Zero angles -> identity rotation",
      torch.allclose(R_zero, torch.eye(3), atol=1e-6))

# -----------------------------------------------------------------------
# 2. params_to_affine validity
# -----------------------------------------------------------------------
print("\n=== 2. params_to_affine properties ===")

translation = torch.zeros(3)
rotation = torch.zeros(3)
scale = torch.ones(3)
shear = torch.zeros(3)
A_id = params_to_affine(translation, rotation, scale, shear)

check("Identity params -> 4x4 identity",
      torch.allclose(A_id, torch.eye(4), atol=1e-6),
      f"\n{A_id}")

translation = torch.tensor([0.1, -0.2, 0.3])
rotation = torch.tensor([0.05, -0.1, 0.08])
scale = torch.tensor([1.1, 0.9, 1.05])
shear = torch.tensor([0.02, -0.01, 0.03])
A = params_to_affine(translation, rotation, scale, shear)

check("Affine is 4x4", A.shape == (4, 4))
check("Last row is [0, 0, 0, 1]",
      torch.allclose(A[3], torch.tensor([0., 0., 0., 1.]), atol=1e-6))

det_3x3 = torch.det(A[:3, :3])
check("3x3 sub-matrix has positive determinant",
      det_3x3.item() > 0,
      f"det = {det_3x3.item():.6f}")
check("Affine is invertible",
      torch.det(A).abs().item() > 1e-6,
      f"det(4x4) = {torch.det(A).item():.6f}")

# -----------------------------------------------------------------------
# 3. affine_to_dense_displacement
# -----------------------------------------------------------------------
print("\n=== 3. affine_to_dense_displacement ===")

size = (16, 16, 16)
A_id_batch = torch.eye(4).unsqueeze(0)  # (1, 4, 4)
disp_id = affine_to_dense_displacement(A_id_batch, size)

check("Identity affine -> zero displacement",
      disp_id.abs().max().item() < 1e-5,
      f"max |disp| = {disp_id.abs().max().item():.6e}")
check("Displacement shape is (B, 3, D, H, W)",
      disp_id.shape == (1, 3, 16, 16, 16),
      f"got {disp_id.shape}")

# Pure translation
A_trans = torch.eye(4).unsqueeze(0).clone()
A_trans[0, 0, 3] = 0.5  # translate x by 0.5 in normalised coords
disp_trans = affine_to_dense_displacement(A_trans, size)
check("Pure translation -> spatially uniform displacement",
      (disp_trans[0, 0].max() - disp_trans[0, 0].min()).item() < 1e-5,
      "x-displacement should be constant across all voxels")
check("Translation amount is correct",
      torch.allclose(disp_trans[0, 0, 0, 0, 0],
                     torch.tensor(0.5), atol=1e-5),
      f"got {disp_trans[0, 0, 0, 0, 0].item():.4f}")

# Batch dimension
A_batch = torch.eye(4).unsqueeze(0).expand(3, -1, -1).clone()
disp_batch = affine_to_dense_displacement(A_batch, size)
check("Batch of 3 -> output batch dim = 3",
      disp_batch.shape[0] == 3)

# -----------------------------------------------------------------------
# 4. AffineNet forward pass
# -----------------------------------------------------------------------
print("\n=== 4. AffineNet forward pass ===")

C = 5
vol_size = (32, 32, 32)
B = 2
affine_net = AffineNet(in_channels=C * 2)

moving = torch.randn(B, C, *vol_size)
fixed = torch.randn(B, C, *vol_size)

matrices = affine_net(moving, fixed)
check("Output shape is (B, 4, 4)", matrices.shape == (B, 4, 4))

for b in range(B):
    M = matrices[b]
    check(f"  Sample {b}: last row is [0,0,0,1]",
          torch.allclose(M[3], torch.tensor([0., 0., 0., 1.]), atol=1e-6))
    det = torch.det(M[:3, :3])
    check(f"  Sample {b}: positive determinant",
          det.item() > 0, f"det = {det.item():.6f}")

# At initialisation (all zeros), should be very close to identity
check("At init, affine is near identity",
      torch.allclose(matrices[0], torch.eye(4), atol=0.05),
      f"max deviation = {(matrices[0] - torch.eye(4)).abs().max().item():.4f}")

# -----------------------------------------------------------------------
# 5. UNet forward pass
# -----------------------------------------------------------------------
print("\n=== 5. UNet forward pass ===")

unet = UNet(in_channels=C * 2, out_channels_flow=3, out_channels_lambda=1)
x = torch.randn(B, C * 2, *vol_size)
flow, lambda_map = unet(x)

check("Flow shape is (B, 3, D, H, W)",
      flow.shape == (B, 3, *vol_size),
      f"got {flow.shape}")
check("Lambda shape is (B, 1, D, H, W)",
      lambda_map.shape == (B, 1, *vol_size),
      f"got {lambda_map.shape}")
check("Lambda is strictly positive (Softplus)",
      (lambda_map > 0).all().item())

# -----------------------------------------------------------------------
# 6. SpatialTransformer
# -----------------------------------------------------------------------
print("\n=== 6. SpatialTransformer ===")

stn = SpatialTransformer(size=vol_size)

zero_flow = torch.zeros(B, 3, *vol_size)
warped = stn(moving, zero_flow)

# With align_corners=False the linspace(-1,1) identity grid doesn't land
# exactly on voxel centres, so border voxels get interpolated.  Check the
# interior (crop 2 voxels on each side) which should be near-exact.
s = 2  # border margin to skip
interior_diff = (warped[:, :, s:-s, s:-s, s:-s] -
                 moving[:, :, s:-s, s:-s, s:-s]).abs().max().item()
check("Zero flow -> interior matches input (crop border)",
      interior_diff < 1e-3,
      f"max interior diff = {interior_diff:.6e}")

check("Output shape matches input",
      warped.shape == moving.shape)

# -----------------------------------------------------------------------
# 7. Full pipeline: Affine -> STN -> UNet -> STN -> Loss
# -----------------------------------------------------------------------
print("\n=== 7. Full pipeline forward + backward ===")

affine_net = AffineNet(in_channels=C * 2)
unet = UNet(in_channels=C * 2, out_channels_flow=3, out_channels_lambda=1)
stn = SpatialTransformer(size=vol_size)

moving = torch.randn(B, C, *vol_size)
fixed = torch.randn(B, C, *vol_size)

# Forward
affine_matrix = affine_net(moving, fixed)
affine_disp = affine_to_dense_displacement(affine_matrix, vol_size)
moving_affine = stn(moving, affine_disp)

x = torch.cat([moving_affine, fixed], dim=1)
flow, lambda_map = unet(x)
total_flow = affine_disp + flow
warped = stn(moving, total_flow)

check("Warped shape matches fixed",
      warped.shape == fixed.shape)

# Loss
loss_weights = {
    "dice": 1.0, "cross_entropy": 0.0,
    "lambda_smoothness": 0.1, "lambda_prior": 0.05, "displacement": 0.01,
}
total_loss, loss_dict = compound_loss(warped, fixed, total_flow, lambda_map, loss_weights)

check("Total loss is scalar", total_loss.dim() == 0)
check("Total loss is finite", torch.isfinite(total_loss).item(),
      f"loss = {total_loss.item()}")
check("All sub-losses present",
      set(loss_dict.keys()) >= {"dice", "lambda_smoothness", "lambda_prior", "displacement"})

# Backward
all_params = list(affine_net.parameters()) + list(unet.parameters())
optimizer = torch.optim.Adam(all_params, lr=1e-4)
optimizer.zero_grad()
total_loss.backward()

affine_grads_ok = all(
    p.grad is not None and torch.isfinite(p.grad).all()
    for p in affine_net.parameters() if p.requires_grad
)
unet_grads_ok = all(
    p.grad is not None and torch.isfinite(p.grad).all()
    for p in unet.parameters() if p.requires_grad
)
check("Gradients flow to AffineNet (all finite)", affine_grads_ok)
check("Gradients flow to UNet (all finite)", unet_grads_ok)

optimizer.step()
check("Optimizer step completes without error", True)

# -----------------------------------------------------------------------
# 8. Affine matrix stays valid after gradient update
# -----------------------------------------------------------------------
print("\n=== 8. Affine validity after optimiser step ===")

matrices_after = affine_net(moving, fixed)
for b in range(B):
    M = matrices_after[b]
    det = torch.det(M[:3, :3])
    check(f"  Sample {b}: det > 0 after update",
          det.item() > 0, f"det = {det.item():.6f}")
    check(f"  Sample {b}: last row unchanged",
          torch.allclose(M[3], torch.tensor([0., 0., 0., 1.]), atol=1e-6))

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"Results:  {PASS} passed,  {FAIL} failed  (total {PASS + FAIL})")
if FAIL == 0:
    print("All tests passed!")
else:
    print("Some tests FAILED -- see above.")
    sys.exit(1)
