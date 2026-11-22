"""Trace mode and the command line of graphwalk.py."""

import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import graphwalk  # noqa: E402
from graphs import POSITIVE, basic_graph, node, ref  # noqa: E402


def chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def png(*chunks: bytes) -> bytes:
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00")
    return graphwalk.PNG_SIGNATURE + chunk(b"IHDR", ihdr) + b"".join(chunks) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def run_cli(*args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = graphwalk.main(list(args))
    return code, out.getvalue(), err.getvalue()


def switch_graph():
    g = basic_graph(model_ref=ref(6), steps=ref(7))
    g["6"] = node("LoraLoader", model=ref(1), clip=ref(1, 1), lora_name="one.safetensors", strength_model=0.5,
                  strength_clip=0.5)
    g["7"] = node("ImpactSwitch", select=2, input1=ref(8), input2=ref(10))
    g["8"] = node("PrimitiveInt", value=10)
    g["10"] = node("PrimitiveInt", value=30)
    g["3"]["inputs"]["text"] = ref(11)
    g["11"] = node("RegexReplace", string="blurry", regex_pattern="blurry", replace="noisy")
    return g


class Trace(unittest.TestCase):
    def test_trace_shows_class_input_and_result_per_field(self):
        info, lines = graphwalk.trace_from_prompt(switch_graph())
        text = "\n".join(lines)
        self.assertEqual(info, graphwalk.extract_from_prompt(switch_graph()))
        for expected in (
            "sampler #9 KSampler",
            "[positive]",
            "· -> #2 CLIPTextEncode[0]",
            "[steps]",
            "· -> #7 ImpactSwitch[0]",
            "(select=2)",
            "input2 -> #10 PrimitiveInt[0] (end)",
            "= 30",
            "· -> #6 LoraLoader[0]",
            "model -> #1 CheckpointLoaderSimple[0] (end)",
            "= LoRA 'one.safetensors' 0.5",
            "= 'SD15/apple_model.safetensors'",
        ):
            self.assertIn(expected, text)
        self.assertNotIn("input1 -> #8", text)  # the branch not taken

    def test_dead_end_and_stale_cache_are_visible(self):
        g = basic_graph(sampler_name=ref(6))
        g["5"] = node("SomethingUnknown")
        g["6"] = node("easy showAnything", anything=ref(5), text="euler")
        _, lines = graphwalk.trace_from_prompt(g)
        text = "\n".join(lines)
        self.assertIn("source not resolved: the cached widget is used", text)
        g = basic_graph()
        g["2"]["inputs"]["text"] = ref(6)
        g["6"] = node("ImpactSwitch", select=4, input1=ref(7))
        _, lines = graphwalk.trace_from_prompt(g)
        self.assertIn("the signal stops here", "\n".join(lines))

    def test_trace_does_not_change_the_result(self):
        g = switch_graph()
        self.assertEqual(graphwalk.trace_from_prompt(g)[0], graphwalk.extract_from_prompt(g))

    def test_no_sampler_gives_no_info(self):
        self.assertEqual(graphwalk.trace_from_prompt({"1": node("CLIPTextEncode", text="x")}), (None, []))


class CommandLine(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def write(self, name, data):
        path = Path(self.dir.name) / name
        path.write_bytes(data)
        return str(path)

    def test_json_file_prints_a1111(self):
        path = self.write("graph.json", json.dumps(basic_graph()).encode())
        code, out, _ = run_cli(path)
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), graphwalk.format_a1111(graphwalk.extract_from_prompt(basic_graph())))
        self.assertTrue(out.startswith(POSITIVE))

    def test_json_wrapped_in_prompt_key(self):
        path = self.write("graph.json", json.dumps({"prompt": basic_graph()}).encode())
        self.assertEqual(run_cli(path)[0], 0)

    def test_png_text_chunks(self):
        graph = json.dumps(basic_graph())
        itxt = b"prompt\x00\x00\x00\x00\x00" + graph.encode()
        ztxt = b"prompt\x00\x00" + zlib.compress(graph.encode())
        itxt_z = b"prompt\x00\x01\x00\x00\x00" + zlib.compress(graph.encode())
        for name, body, kind in (
            ("text.png", b"prompt\x00" + graph.encode("latin-1"), b"tEXt"),
            ("itxt.png", itxt, b"iTXt"),
            ("ztxt.png", ztxt, b"zTXt"),
            ("itxt_z.png", itxt_z, b"iTXt"),
        ):
            data = png(chunk(b"tEXt", b"workflow\x00{}"), chunk(kind, body))
            code, out, _ = run_cli(self.write(name, data))
            self.assertEqual(code, 0, name)
            self.assertIn("Steps: 20", out, name)

    def test_png_utf8_text(self):
        graph = basic_graph()
        graph["2"]["inputs"]["text"] = "a red apple · 1152×1728"
        body = b"prompt\x00\x00\x00\x00\x00" + json.dumps(graph, ensure_ascii=False).encode("utf-8")
        code, out, _ = run_cli(self.write("utf8.png", png(chunk(b"iTXt", body))))
        self.assertEqual(code, 0)
        self.assertIn("1152×1728", out)

    def test_trace_flag(self):
        path = self.write("graph.json", json.dumps(basic_graph()).encode())
        code, out, _ = run_cli(path, "--trace")
        self.assertEqual(code, 0)
        self.assertIn("[positive]", out)
        self.assertIn("CLIPTextEncode", out)
        code, out2, _ = run_cli("--trace", path)
        self.assertEqual(out, out2)

    def test_errors(self):
        self.assertEqual(run_cli()[0], 2)
        self.assertEqual(run_cli("a", "b")[0], 2)
        self.assertEqual(run_cli(self.write("bad.json", b"not json"))[0], 2)
        self.assertEqual(run_cli(str(Path(self.dir.name) / "missing.png"))[0], 2)
        no_prompt = self.write("plain.png", png(chunk(b"tEXt", b"other\x00x")))
        self.assertEqual(run_cli(no_prompt)[0], 1)
        no_sampler = self.write("nosampler.json", json.dumps({"1": node("CLIPTextEncode", text="x")}).encode())
        self.assertEqual(run_cli(no_sampler)[0], 1)


if __name__ == "__main__":
    unittest.main()
