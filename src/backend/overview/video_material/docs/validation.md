# Validation record

## Automated verification

`npm test` exercises the following eleven checks:

1. The five ontology source fingerprints match the repository.
2. All five cases are distinct, every quoted span occurs verbatim, and all 30 criterion paths and 40 candidate paths are valid leaves.
3. The full catalogue retains all 7,265 leaves; sampled graph edges retain actual parent-child lineage.
4. The selector obeys domain coverage, round-robin allocation, score refinement and budget limits without duplicate paths.
5. Comparison counts derive from the selected sets and change with the comparison policy.
6. Sparse default cases receive no invented personal network-impact anchor.
7. Evidence weights are complementary and remain within the source bounds.
8. Simulated cycle changes can move personal anchors in either direction.
9. The 174-second, 16-chapter timeline has no gaps and includes the full closed loop.
10. Every case renders at three moments in every scene, plus alternate readiness states, without exceptions or text outside the frame.
11. Returning to an earlier time after another case produces the same pixel buffer.

All eleven checks passed during the rebuild. They verify the demonstration's data and mechanics, not clinical validity or efficacy.

## Visual and interaction review

- Reviewed all five 16-frame contact sheets and representative Full HD frames for the complaint, hierarchy, ancestry, HAPA delivery and closing layouts.
- Corrected native font weight handling by including static font instances, improving legibility and matching the browser's typography.
- Adjusted hierarchy label placement to avoid label collisions.
- Exercised case selection, scene navigation, readiness adjustment, comparison policy, full-catalogue detail, root-branch filtering, zoom, visible depth and candidate inspection in the browser.
- Verified the full-catalogue readout reaches 7,265 of 7,265 leaves and the SOCIAL-only readout reaches 1,559 of 1,559.
- Checked presentation mode and keyboard entry/exit, and inspected the small-window responsive presentation.
- Rendered a custom saved-session preview with alternate case, readiness, cycle, depth, density and motion settings.
- Checked local documentation links and the absence of em dashes in authored source and documentation.

The in-app browser's viewport override did not provide the requested large viewport geometry in this session. Full HD composition was therefore inspected from the deterministic exported frames, while live interaction and the narrow layout were verified in the browser. The responsive desktop CSS is included; this record does not claim a browser screenshot at a viewport that was not actually obtained.

## Media verification

Run:

```sh
npm run verify:media
```

The media verifier inspects all five complete MP4s with `ffprobe`, checks H.264, 1920 × 1080, 24 fps, 4,176 frames, 174 seconds and `yuv420p`, then decodes each entire film with FFmpeg. It also verifies that every poster and storyboard companion exists and that each film fits ordinary GitHub file limits.

All five films passed the full metadata and decode checks. Each contains 4,176 frames and is approximately 10.6 to 10.8 MiB.

Machine-readable output is saved in [renders/verification.json](../renders/verification.json). Rendering metadata is saved separately in [renders/manifest.json](../renders/manifest.json).
