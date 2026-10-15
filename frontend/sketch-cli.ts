// Backend entry to the sketch solver: one JSON sketch on stdin, {sketch, report} on stdout.
// Run as `node sketch-cli.ts` (Node runs TypeScript directly). Used by backend/src/cad_assist/sketch/solve.py.
import { solveSketch } from './src/sketch/solve.ts'

let input = ''
process.stdin.setEncoding('utf8')
process.stdin.on('data', (d) => (input += d))
process.stdin.on('end', async () => {
  try {
    const out = await solveSketch(JSON.parse(input))
    process.stdout.write('\n@@SKETCH@@' + JSON.stringify(out))
  } catch (e) {
    process.stdout.write('\n@@SKETCH@@' + JSON.stringify({ error: String(e instanceof Error ? e.message : e) }))
    process.exitCode = 1
  }
})
