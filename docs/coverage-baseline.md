# Coverage Baseline

Recorded: 2026-07-16
Command: `bash scripts/coverage.sh`

## Summary

- Combined total: 63%
- Total tests: 205 passed, 1 deselected
- CI global floor target: `fail_under=68` in todo 8; this file records the measured baseline used for that ratchet

## Worst-Covered Modules (bottom 10)

| Module | Coverage | Notes |
|--------|----------|-------|
| nanovllm_voxcpm/lora_ops/triton_ops/kernel_utils.py | 7% | Triton GPU kernels |
| nanovllm_voxcpm/models/voxcpm2/utils.py | 9% | Pure helpers |
| nanovllm_voxcpm/models/voxcpm/model.py | 12% | GPU-coupled forward passes |
| nanovllm_voxcpm/layers/attention.py | 20% | Mixed kernel logic |
| nanovllm_voxcpm/layers/embed_head.py | 25% | CPU-testable logic |
| nanovllm_voxcpm/models/voxcpm/runner.py | 26% | Orchestration logic |
| nanovllm_voxcpm/layers/audio_vae.py | 28% | GPU-coupled VAE |
| nanovllm_voxcpm/lora_ops/triton_ops/lora_shrink_op.py | 37% | Triton GPU kernels |
| nanovllm_voxcpm/__init__.py | 40% | Import-only package surface |
| nanovllm_voxcpm/models/voxcpm2/model.py | 44% | GPU-coupled model logic |

## Full Per-Module Table

| Name | Stmts | Miss | Branch | BrPart | Cover | Missing |
|------|------:|-----:|-------:|-------:|------:|---------|
| deployment/app/api/routes/generate.py | 140 | 12 | 60 | 6 | 90% | 170-171, 195, 198-199, 208, 216->220, 221, 227-228, 231-233, 247->250 |
| deployment/app/core/config.py | 165 | 4 | 56 | 4 | 96% | 182, 186, 190, 231 |
| deployment/app/core/lifespan.py | 46 | 2 | 12 | 4 | 90% | 24, 29, 38->42, 67->exit |
| deployment/app/core/metrics.py | 55 | 14 | 4 | 1 | 71% | 90-106 |
| deployment/app/services/mp3.py | 118 | 9 | 26 | 2 | 92% | 43->39, 108->112, 119-120, 136, 155-156, 159-160, 165-166 |
| nanovllm_voxcpm/__init__.py | 10 | 6 | 0 | 0 | 40% | 5-11 |
| nanovllm_voxcpm/engine/block_manager.py | 94 | 2 | 26 | 3 | 96% | 108, 146, 173->exit |
| nanovllm_voxcpm/engine/llm_engine.py | 105 | 4 | 20 | 1 | 96% | 155-156, 186-187, 205->204 |
| nanovllm_voxcpm/engine/lora_manager.py | 298 | 29 | 90 | 21 | 87% | 134-135, 202, 205, 212, 218, 221-222, 244-245, 252->exit, 261, 266, 269, 279, 282, 287->exit, 310-313, 324-326, 378, 380, 418, 432, 439, 449, 452 |
| nanovllm_voxcpm/engine/model_runner.py | 499 | 218 | 150 | 22 | 53% | 124-130, 187, 209-267, 292, 295-315, 331->334, 341, 343, 353, 371, 374-375, 378, 381, 384, 387, 390-420, 424-494, 497-550, 558-566, 575-576, 610->612, 617-618, 626-627, 632, 639, 641, 644-646, 649-694, 697-716, 782->789, 790->796, 847->853, 851, 873, 879, 885->912, 905, 910, 948, 960 |
| nanovllm_voxcpm/engine/scheduler.py | 91 | 0 | 34 | 3 | 98% | 101->103, 130->114, 168->exit |
| nanovllm_voxcpm/engine/sequence.py | 45 | 1 | 0 | 0 | 98% | 110 |
| nanovllm_voxcpm/layers/attention.py | 49 | 37 | 10 | 0 | 20% | 21-31, 41-47, 60-66, 69-102 |
| nanovllm_voxcpm/layers/audio_vae.py | 156 | 106 | 20 | 0 | 28% | 12, 16, 21-22, 25-26, 31-33, 36, 40, 44, 50-54, 59-60, 63, 67-70, 75-77, 92-97, 102-104, 119, 130-148, 151-152, 161-162, 165-170, 182-203, 206, 211, 224-263, 266, 285-315, 318-326, 346, 357-361 |
| nanovllm_voxcpm/layers/audio_vae_v2.py | 226 | 68 | 54 | 8 | 64% | 12, 16, 26-27, 37, 50-54, 63, 85-90, 113, 137-138, 147-148, 151-153, 178, 190, 211-225, 237-243, 316, 320-329, 351, 353, 367, 392, 394, 402, 406-409 |
| nanovllm_voxcpm/layers/embed_head.py | 50 | 36 | 6 | 0 | 25% | 17-26, 29-33, 36-44, 54-55, 58-69 |
| nanovllm_voxcpm/layers/linear.py | 97 | 49 | 8 | 1 | 47% | 31-32, 55, 58, 61, 83, 93-94, 97-104, 116-122, 125-139, 149-150, 153-158, 161-164 |
| nanovllm_voxcpm/layers/lora.py | 567 | 236 | 230 | 32 | 52% | 36, 42, 73, 76, 80, 87, 89, 136, 142-159, 162-168, 176, 178, 180, 186, 281-282, 288->exit, 290->293, 293->296, 296->exit, 307-308, 323-326, 331-338, 369-382, 385-393, 428, 430, 433-448, 453-457, 460-465, 499-500, 506->exit, 519-525, 529-532, 537-544, 551, 561-571, 589-597, 624-628, 631-636, 645-665, 680-705, 708-712, 715-720, 723-742, 752-762, 765-767, 770-774, 783-797, 872->exit, 878, 892, 894, 896, 898, 900, 904, 917 |
| nanovllm_voxcpm/layers/rotary_embedding.py | 32 | 3 | 0 | 0 | 91% | 59-61 |
| nanovllm_voxcpm/llm.py | 43 | 2 | 16 | 1 | 95% | 11-12, 33->41 |
| nanovllm_voxcpm/lora.py | 211 | 56 | 72 | 16 | 70% | 101, 104, 118, 141-178, 182, 185-186, 265, 267, 269, 272, 274, 289, 297, 313, 317, 341, 345, 369-380, 384-389, 392-399, 403, 410, 418->420, 439 |
| nanovllm_voxcpm/lora_ops/triton_ops/kernel_utils.py | 93 | 84 | 32 | 0 | 7% | 25-66, 99-140, 174-228 |
| nanovllm_voxcpm/lora_ops/triton_ops/lora_expand_op.py | 48 | 22 | 8 | 2 | 50% | 43-62, 110, 133 |
| nanovllm_voxcpm/lora_ops/triton_ops/lora_kernel_metadata.py | 48 | 4 | 6 | 2 | 85% | 52, 62-64 |
| nanovllm_voxcpm/lora_ops/triton_ops/lora_shrink_op.py | 137 | 86 | 30 | 6 | 37% | 47-73, 127-173, 187-189, 191-193, 195-197, 199-201, 272, 295-309 |
| nanovllm_voxcpm/lora_ops/triton_ops/utils.py | 82 | 12 | 20 | 5 | 83% | 26-27, 40, 66-67, 99-103, 130-131 |
| nanovllm_voxcpm/models/voxcpm2/engine.py | 106 | 37 | 28 | 8 | 62% | 32-42, 45-46, 49-72, 99->101, 105, 112, 117, 142-160, 163-170, 210->212, 212->215 |
| nanovllm_voxcpm/models/voxcpm2/lora_loader.py | 175 | 20 | 80 | 20 | 84% | 34, 40, 46, 50, 52, 59, 69, 72, 82, 84, 87, 90, 92, 107, 115->117, 126, 142, 167-168, 192, 205 |
| nanovllm_voxcpm/models/voxcpm2/model.py | 350 | 187 | 40 | 2 | 44% | 39-50, 53-65, 68-75, 80-87, 91, 120-179, 182-208, 219-252, 255, 267-291, 294-300, 312-323, 326-330, 339-345, 357, 373, 397-408, 436-445, 479->481, 503-509, 520-522, 649-684 |
| nanovllm_voxcpm/models/voxcpm2/runner.py | 98 | 20 | 12 | 3 | 79% | 41-45, 52-59, 62, 72, 78-80, 125, 130-131, 149->153 |
| nanovllm_voxcpm/models/voxcpm2/server.py | 391 | 154 | 106 | 23 | 56% | 54, 92, 108->110, 111, 130-154, 172-173, 176-177, 180, 183, 186, 198-199, 200->203, 205-207, 224-225, 231->243, 236, 239-240, 244-258, 276, 322-323, 326-327, 335->337, 339-341, 344-345, 347->331, 349->331, 353, 370, 377-383, 387, 391->398, 395-396, 404, 406-407, 409-412, 415, 422, 425, 428, 442-468, 485-504, 507, 510, 513-514, 517-519, 522-525, 528, 532, 538-542, 548, 550, 579-587, 626-642, 645-648, 651-652, 655-656, 659-660, 663-664, 667-668, 671-672, 675-676, 691-708 |
| nanovllm_voxcpm/models/voxcpm2/utils.py | 26 | 23 | 6 | 0 | 9% | 7-37 |
| nanovllm_voxcpm/models/voxcpm/engine.py | 97 | 33 | 28 | 5 | 62% | 35-46, 49-50, 53-82, 110->113, 117, 124, 129, 156-165, 218-223 |
| nanovllm_voxcpm/models/voxcpm/lora_loader.py | 183 | 24 | 84 | 22 | 83% | 35, 41, 47, 51, 53, 61, 71, 74, 84, 86, 89, 92, 94, 109, 117->119, 128, 144, 157, 173-174, 199, 212, 255-257 |
| nanovllm_voxcpm/models/voxcpm/model.py | 427 | 370 | 54 | 0 | 12% | 39-40, 48-55, 71-88, 91-107, 118-133, 140-153, 164-173, 193-273, 280-328, 339-386, 389-392, 403-429, 438-448, 459-473, 480-485, 490-492, 495-503, 513-522, 525-528, 542-572, 590-608, 621-631, 656-669, 672-676, 697-736, 741-747, 753-762, 767-774, 777-781, 799-897, 916-971 |
| nanovllm_voxcpm/models/voxcpm/runner.py | 107 | 75 | 18 | 0 | 26% | 45-49, 53, 56-68, 71, 81-91, 97-99, 109-203 |
| nanovllm_voxcpm/models/voxcpm/server.py | 391 | 160 | 106 | 23 | 56% | 52, 93, 108, 111, 131-155, 168-169, 172-173, 176, 179, 182, 197-198, 199->202, 204-211, 240-242, 253->265, 258, 261-262, 267-292, 316, 368-369, 372-373, 382->384, 386-388, 391-395, 398->377, 402-404, 421, 430-434, 439, 445->453, 449-451, 462, 465-466, 470-473, 480, 487, 490, 493, 506-533, 550-572, 575, 578, 582-583, 587-589, 593, 600-604, 611, 613, 630-636, 639, 684-693, 726-742, 745-748, 751-752, 755-756, 759-760, 763-764, 767-768, 771-772, 775-776, 813-830 |
| nanovllm_voxcpm/models/voxcpm/utils.py | 28 | 3 | 6 | 1 | 88% | 61, 100-101 |
| nanovllm_voxcpm/utils/context.py | 69 | 3 | 8 | 3 | 92% | 81, 91, 150 |
| nanovllm_voxcpm/utils/distributed.py | 9 | 2 | 4 | 2 | 69% | 7, 13 |
| nanovllm_voxcpm/utils/loader.py | 39 | 2 | 16 | 0 | 96% | 12-13 |
| TOTAL | 6462 | 2215 | 1596 | 252 | 63% |  |

30 files skipped due to complete coverage.
Coverage HTML written to dir htmlcov
Coverage XML written to file coverage.xml

## Notes

- `deployment/client.py` is omitted from coverage because it is a demo entrypoint requiring a live server.
- GPU-only paths (Triton kernels, flash_attn calls, model forward passes) are intentionally excluded with `@pytest.mark.gpu` and `# pragma: no cover`.
- This document is the committed baseline referenced by todo 8 for the combined coverage ratchet toward `fail_under=68`.
