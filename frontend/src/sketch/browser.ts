// The solver in the browser: Vite serves the .wasm, the shared module does the rest.
import wasmUrl from '@salusoft89/planegcs/dist/planegcs_dist/planegcs.wasm?url'
import { loadSolver } from './solve.ts'

export const browserSolver = () => loadSolver({ locateFile: () => wasmUrl })
