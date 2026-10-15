# CLAUDE.md — 3D Print Assist

## What this is
An AI-native CAD tool. The user (an aerospace stress engineer and design manager, CATIA user) is the product owner and beta tester; Claude writes nearly all of the code. The plan, milestones, decisions log and design-rule table live in the Claude Doc linked in `README.md`. Keep its decisions log current when a decision changes.

## Core architecture (do not drift from these)
- **Intent model is the source of truth.** Interfaces, keep-outs, loads and rules are typed intent objects with local frames; every pillar (CAD, FEM, DFM, GD&T) reads them.
- **Parts are built from the parametric feature library** (plate, bolt pattern, lug, clevis, gusset, pocket, fillet, chamfer…). The AI composes the same features a user edits. One feature definition generates its UI, AI tool description and rule checks.
- **Named faces, not face indices,** for anything downstream refers to. Picked faces on imported parts are matched by geometric signature (type, area, centroid, normal / axis / radius).
- **No custom geometry kernel.** build123d on OpenCascade (OCP).

## Conventions
- Geometry is held in **millimetres** internally (OCCT native); the UI displays inches by default. Convert only at the edges (`frontend/src/units.ts`, `build123d.IN`).
- Z is up. Mouse bindings default to CATIA (middle-drag pan, middle + left rotate, middle + click-left zoom).
- Backend: Python 3.12, uv, FastAPI, pytest. Every geometry function gets a test with a hand-checked number (volume, area, position).
- Frontend: React + TypeScript + plain three.js (no react-three-fiber). `erasableSyntaxOnly` is on, so no constructor parameter properties or enums.
- Use only non-proprietary geometry and data. Sample mating parts come from `backend/src/print_assist/geometry/samples.py`.

## AI loop
- `backend/src/print_assist/ai/generate.py`: Claude proposes feature trees via tools; the backend builds and validates each one. Claude never writes geometry. Locked features are restored server-side whatever Claude proposes.
- The API key lives in `backend/.env` (never committed). Tests use a scripted fake client; never call the real API from tests.

## Topology optimisation
- `backend/src/print_assist/topo/`: `fe.py` (voxel hex8 elements, 3 translational DOFs per node; stiffness pattern precomputed, reassembly is vector adds; pyamg smoothed aggregation with the 6 rigid-body modes, sparse coarse solve), `simp.py` (min compliance, density filter, OC), `setup.py` (intent → grid, element size iterated to fit the element budget), `job.py` (sweep runner, results saved under `<project>/topo/`), `machining.py` (machining set-up), `prismatic.py` (2.5D parametrisation: per-face depth-layer maps, cover rule between layers, intersection of faces; gradient tested against finite differences). Compare raw and machined results as 0/1 parts at equal mass (`solid_compliance`), never on grey densities.
- `topo/interpret.py` reads a machinable result as a plate (pins, floors, cut-outs, exact clearances by ray casting, lug rule) and runs the "Make parametric model" job; `topo/sketch_seed.py` drafts the first sketches from that reading. The voxel tracer is for measuring, not for the model.
- The topology mesh is for layout only (full-integration bricks are stiff in bending); stresses come from FEA on the parametric model later.

## Sketches
- `backend/src/print_assist/sketch/` (model, solve, geometry, from_faces) mirrors `frontend/src/sketch/` (model, solve). One solver: planegcs as WebAssembly, run in the browser and by the backend through `node frontend/sketch-cli.ts` (Node runs the TypeScript directly). Keep the two models in step.
- Tangency at a shared endpoint is an angle-via-point constraint (as FreeCAD does); edge tangency plus the shared point reads as redundant. Infer no constraint the others already imply (horizontal on a tangent line, the fourth right angle of a rectangle).
- Splines are Catmull–Rom through their points, built as the same Bezier segments on both sides (`draw.ts` splineControls / `geometry.py` spline_edges); `tangent_start` / `tangent_end` make an end follow the curve it joins. The solver moves the points only.
- The sketcher (`frontend/src/SketchEditor.tsx`, edits in `sketch/edit.ts`) solves live with temporary drag constraints and commits each finished edit with `PUT /api/project/design/sketches/{id}`.
- Pad / sketch_pocket consume sketches; nested closed profiles alternate material and hole. Sketch points tied to an intent object (`ref`, a pin centre) must match it or the feature errors until re-solved.

## Stress analysis (FEA)
- `backend/src/print_assist/fem/`: `element.py` (C3D10, Gmsh node order, 4-point Gauss, stresses extrapolated and averaged; 6-node triangle tractions), `mesh.py` (Gmsh in a spawned child process: its high-order optimiser can abort the process; curved, then optimised, then straight-sided, inverted-element check), `solve.py` (AMG-CG, rigid-body near-nullspace), `analysis.py` (reactions from in-plane statics, cosine bearing with exact resultants, 3-2-1 hold whose reactions must be ~0, regions, margins), `job.py` (background job, results saved to `<project>/fem/`), `sizing.py` (recipe-level sizing for a target margin).
- Mesh controls (`fem/controls.py`, project `mesh`): global size / curvature, edge divisions, mapped faces, spheres of influence. Edges and faces are kept by centre of mass and size (re-found on a rebuilt part and inside Gmsh), not by index. `propose()` is the best-practice rules; `ai/mesh_plan.py` has Claude propose controls, meshing and measuring each (elements around / through each bore, scaled-Jacobian quality, DOF and memory). A checked mesh is reused by the solve when its key (controls + part) matches. The convergence check re-solves with every size x1.5.
- Pin contact (`fem/contact.py`, `analyse(..., bearing="contact")`): a meshed steel pin per joint (bore minus 0.0005 in clearance) across the fork gap and into the tines; each fork grips its pin's stubs as a rigid body that holds it square (pivot fixed, link held along the link, load fork free in the plane with the force); frictionless penalty contact at the bore's Gauss points, quadratic on both sides, active-set passes (loose CG while the set moves, tight to confirm). Pin forces from contact must equal statics. A contact run also solves the cosine model on the same mesh and reports contact / cosine per region (`calibration`) for the fast model. Benchmarks: link equilibrium and bearing arc, Hertz within 3%, offset lugs.
- Converged peaks (`fem/converge.py`, job option `peaks`): hot spots are local von Mises peaks within 10% of the governing one; each round adds a sphere of influence per spot at its measured local size / 1.5 (Gmsh shortfall pushed into the next target, divided edges inside raised to match) and re-solves. A spot is converged when its peak is within 5% of the last mesh at least 1.3x coarser there; singular when the peak keeps rising and the change is not shrinking (ratio >= 0.7; smooth quadratic peaks give ~0.44). Contact runs converge with the cosine model and carry each spot's contact / cosine factor (or with contact every round, `peaks_contact`). Benchmarks: hole in a plate converges on Kirsch, sharp L corner flagged singular.
- The solver reports progress (stage, CG residual per iteration against rtol) through `monitor` into `FemJob.progress` for the solve-status window. `FemRunner.job` is the latest job of any kind; `results_job` the latest analysis or sizing.
- Every new analysis capability gets a closed-form benchmark in `tests/test_fem.py` before it is trusted. Peaks are taken over every node (mid-edge nodes included) so they bound the contour.
- Allowables: MIL-HDBK-5J (public) A-basis LT values in `intent/models.py` MATERIAL_PROPS, overridable in the rules; MMPDS is licensed and not used.

## Running
- `./dev.sh` starts both servers (backend :8000, frontend :5173).
- Tests: `cd backend && uv run pytest`; `cd frontend && npm run build`.

## Deployment
Local while iterating; stable builds go on the user's home server. Plan placement with the `homelab-consultant` agent and follow the homelab repo's conventions (`~/Projects/homelab`).
