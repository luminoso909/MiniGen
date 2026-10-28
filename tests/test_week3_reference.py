"""Test only the tagged, authorized teaching core; never install it into src."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def reference():
    path = Path("docs/learning/S2_第3周_图标数据与潜空间DiT基础.ipynb")
    notebook = json.loads(path.read_text())
    cells = [
        c for c in notebook["cells"] if "dit_reference" in c.get("metadata", {}).get("tags", [])
    ]
    assert len(cells) == 1
    namespace = {}
    exec(compile("".join(cells[0]["source"]), str(path), "exec"), namespace)
    return namespace


def test_reference_passes_independent_learner_contract(reference):
    spec = importlib.util.spec_from_file_location("checks", "scripts/check_dit_core.py")
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    checks.verify(
        reference["patchify_reference"],
        reference["unpatchify_reference"],
        reference["TeachingDiTBlock"],
    )


def test_patch_mapping_gradient_and_noncontiguous_shapes(reference):
    patch = reference["patchify_reference"]
    unpatch = reference["unpatchify_reference"]
    image = torch.randn(2, 3, 4, 6, dtype=torch.float64, requires_grad=True)
    tokens = patch(image, 2)
    for b in range(2):
        for r in range(2):
            for s in range(3):
                for i in range(2):
                    for j in range(2):
                        for c in range(3):
                            assert (
                                tokens[b, r * 3 + s, (i * 2 + j) * 3 + c]
                                == image[b, c, r * 2 + i, s * 2 + j]
                            )
    assert torch.autograd.gradcheck(lambda x: patch(x, 2), (image,))
    transposed = image.transpose(-1, -2)
    torch.testing.assert_close(unpatch(patch(transposed, 2), 3, 6, 4, 2), transposed)
    with pytest.raises(ValueError):
        unpatch(tokens, 4, 4, 6, 2)


def test_attention_matches_independent_torch_kernel(reference):
    torch.manual_seed(8)
    attn = reference["TeachingAttention"](16, 4).double()
    x = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
    q, k, v = attn.qkv(x).reshape(2, 5, 3, 4, 4).permute(2, 0, 3, 1, 4).unbind(0)
    independent = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    expected = attn.projection(independent.transpose(1, 2).reshape(2, 5, 16))
    torch.testing.assert_close(attn(x), expected)
    assert torch.autograd.gradcheck(attn, (x,))


def test_zero_head_gradient_stages_and_condition_broadcast(reference):
    torch.manual_seed(7)
    model = reference["TeachingDiT"](depth=1, hidden=32)
    z, t = torch.randn(2, 4, 16, 16), torch.tensor([1, 700])
    target = torch.randn_like(z)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    stages = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(z, t)
        if step == 0:
            assert torch.count_nonzero(prediction) == 0
        (prediction - target).square().mean().backward()
        block = model.blocks[0]
        stages.append(
            (
                block.modulation[-1].weight.grad.norm().item(),
                block.attention.qkv.weight.grad.norm().item(),
            )
        )
        optimizer.step()
    assert stages[0] == (0, 0)
    assert stages[1][0] > 0 and stages[1][1] == 0 and stages[2][1] > 0
    block = model.blocks[0]
    with pytest.raises(ValueError):
        block(torch.randn(2, 8, 32), torch.randn(1, 32))
