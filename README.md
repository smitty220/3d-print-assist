# CAD Assist

An AI-native engineering design tool: define a part's interfaces, keep-outs and loads, and iterate with an AI toward a stress-checked, machinable, toleranced part. Phase 0 is the constraint-driven CAD generator for machined aluminum brackets and fittings.

Plan and roadmap: [CAD Assist — Vision & Roadmap](https://claude.ai/code/artifact/da1e3ca5-444a-4d1f-abc9-eb1b91569cd9)

## Run locally

Requires [uv](https://docs.astral.sh/uv/) and Node 23.6+ (the backend runs the TypeScript sketch solver through Node).

```sh
./dev.sh
```

Then open http://127.0.0.1:5173. The backend serves on port 8000 and the frontend proxies `/api` to it.

## Layout

| Path | What it is |
| --- | --- |
| `backend/` | FastAPI app, build123d / OpenCascade geometry, pytest suite |
| `backend/src/cad_assist/geometry/` | STEP I/O, face-tagged tessellation, sample mating parts |
| `frontend/` | React + TypeScript + three.js viewer |
| `backend/src/cad_assist/intent/` | Intent model and builders (interfaces from picked faces) |
| `backend/src/cad_assist/features/` | Feature library, design build pipeline, reference-bracket templates |
| `backend/src/cad_assist/topo/` | Topology optimisation: voxel FE (8-node bricks, AMG-preconditioned CG), SIMP optimiser, intent-to-grid setup, background sweep runner |
| `backend/src/cad_assist/sketch/` | 2D sketches (points, lines, arcs, circles, splines, constraints): model, solve bridge, sketch → faces, faces → constrained sketch |
| `frontend/src/sketch/` | Sketch model and solver (planegcs, FreeCAD's solver as WebAssembly), shared with the backend via `frontend/sketch-cli.ts` |
| `backend/src/cad_assist/fem/` | Stress analysis: Gmsh tet10 meshing (in a child process), C3D10 elements, AMG-CG solve, pin loads as cosine bearing, statics for reactions, margins, sizing |
| `frontend/src/viewer/Viewer.ts` | Viewport: CATIA navigation, per-face picking, section tool |

## Tests

```sh
cd backend && uv run pytest
cd frontend && npm run build   # type-check + build
```

## Status

- **M0.1 (done):** STEP import, face-tagged meshes with geometric signatures, clickable faces, CATIA navigation.
- **M0.2 (done):** intent model and editor; picked faces become planar mounts, bolt patterns and bores; keep-out volumes; fastener catalog with 3D hardware; design space, load cases and design rules; interfaces survive part revisions.
- **M0.3 (done):** parametric feature library (plate, wall, bolt holes, clevis, lug, gusset pair, pocket, fillet, chamfer) with intent-driven parameters, rule checks and named faces; three reference brackets (clevis, single-lug fitting, angle bracket); feature tree with live editing and locks.
- **M0.4 (done):** validator. Pre-flight on the intent (mount, design space, fastener fit, pin position, interfaces inside the space, keep-out vs interfaces, loads, mass target) and part checks on every rebuild (build, rule checks, keep-out clearance, design-space containment, interference with mating parts, mount seating and overhang, bolt holes on pattern with head/washer clearance, pin bores coaxial with a clear pin path, sampled minimum wall, mass). Validation panel with viewport markers.
- **M0.5 (in progress):** AI generation loop. Claude composes library features through tools (get a template, build and check, finish); every candidate is built and validated, shown live, and the best is kept; locked features are enforced; one-click undo. Gated by pre-flight. Offline replay mode for UI work. Remaining: first runs against the real API and prompt tuning on the reference cases. Paused while topology optimisation comes first.
- **T1–T3 (in progress):** topology optimisation as the first design step. Interfaces get supports (bolt patterns on the mount Fixed by default; mount face Normal as an option). The design space is meshed with cubic 8-node bricks sized to an element budget; mating parts, keep-outs (plus clearance), bolt shank and head-access columns and the pin path are left out, and seat and lug rings are held solid. Loads are spread over the bore (bearing side) or the clamped area, with exact resultant force and moment. SIMP minimum compliance (max stiffness for a given mass), density filter, OC updates; one run at the mass target by default (35 s on the clevis at 60k elements, 13 s at 20k), with an optional three-step sweep (×1.5, ×1, ×0.65) warm-started high to low. Viewer shows the set-up, the run live, and every result with a mass slider and density cutoff (section caps work on it). Next: AI interpretation of the result into a parametric model.
- **T4 (done):** 2.5D machinable topology optimisation for a 4-axis mill. Each setup face (the rotary indexed every 90° about the long axis, plus an optional open end) gets a few flat floor depths; each depth layer is a 2-D map extruded along the cutter, a pocket can only deepen under cut-away layers, and the part is the intersection of the faces, so walls are parallel to the cutter and floors flat by construction. A machinable run finds the raw optimum first, fits the layers to it and re-optimises; both are compared as solid parts at the same mass. Clevis, rotary about Z, 3 depths: 8.7% less stiff than raw at 0.36 lb, 63 s for both at 60k elements.
- **Bell crank example (done):** the default project is now a bell crank (pivot fitting, input rod end, output link). New bore supports: Pin (held, free to turn) and Link (reacts along the rod only), as bearing springs; the link takes exactly 1500 × 4.0/2.5 lbf. The clevis project is kept in `backend/data/projects/clevis_bracket` (`CAD_ASSIST_PROJECT=clevis_bracket`).
- **T5 S1 (done):** sketch-based 2.5D model from a topology result. Sketches with constraints and dimensions, solved by planegcs (the same WebAssembly solver in the browser and, via Node, the backend); Pad and Pocket features consume them (nested profiles are holes or islands). "Make parametric model" reads the machinable result (thickness floors, pins, cut-outs, exact clearance to mating parts by ray casting, lug e/D grown to 1.5) and drafts sketches: lug arcs with tangent sides, filleted lightening holes, web pockets inside a rim with bosses, spot-face steps so each lug fits its fork. Live step log in the Topology panel, part built feature by feature over a ghost of the result, mass/stiffness compared on the same mesh. Bell crank: 0.41 lb, as stiff as the 2.5D result, no failing checks, 8 s. 
- **T5 S2 (done):** the sketcher. Click a sketch in the feature tree (or Edit sketch on a pad/pocket) for a 2D view square to the sketch plane over the topology thickness map. Drag points, lines (both ends) and circle edges with the solver running live in the browser; each finished edit is solved again by the backend, saved and rebuilt. Constraint glyphs (tangent, perpendicular, parallel, H/V, concentric, equal) and dimensions on the geometry, thinned out on curves too small to read; click a dimension to type a value; select items for the constraints that fit them; Delete removes; split a line, join curves into a line or a spline (spline ends can follow the curves they join), construction toggle, undo. Degrees of freedom, conflicting and redundant constraints are shown and highlighted. 
- **T5 S3 (done):** Draft with AI: Claude edits a plate recipe (lug radii; outline edges tangent / arc / spline; holes; web; spot faces), compiled into the same constrained sketches; each build validated and compared with the topology result (mass, stiffness, a material map); closing questions you answer in the panel; optional stress check on every build.
- **T6 (done, first version):** stress analysis of the part. Gmsh second-order tets (curved), C3D10 solver verified against closed form (patch test exact; cantilever within 1% of Timoshenko; plate with hole Kt 3.11 vs 3.05 thin-plate), pin forces from statics as cosine bearing pressure, peaks by region (each lug, web/rim) and margins (limit vs Fty, ultimate vs Ftu; MIL-HDBK-5J A-basis allowables, editable), convergence check, contours (von Mises, principal, displacement) and deformed shape, results saved with the project. Sizing finds the lightest drafted part meeting a target margin (bell crank: 0.404 to 0.308 lb at MS +0.12). Next: classical lug checks (AFFDL-TR-69-42) alongside the FEA, pin contact, more load paths (bolted fittings).

### AI setup
Put your key in `backend/.env` (git-ignored); it is re-read on each request, so no restart is needed:

```
ANTHROPIC_API_KEY=sk-ant-...
# optional: CAD_ASSIST_MODEL=claude-opus-5-5
```

Run the backend with `CAD_ASSIST_AI_REPLAY=1` to exercise the loop offline with a scripted stand-in (no key, no cost).
