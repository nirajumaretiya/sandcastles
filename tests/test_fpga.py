"""Tests for the FPGA targeting module."""

import pytest

from w2s.fpga import (
    estimate_fpga,
    generate_build_script,
    generate_constraints,
    ICE40_UP5K,
    ECP5_25K,
    DEVICES,
)


class TestFPGAEstimate:
    def test_returns_estimate(self, xor_quantized_graph):
        report = estimate_fpga(xor_quantized_graph, ICE40_UP5K, "combinational")
        assert report.lut4s_used > 0

    def test_sequential_uses_bram(self, xor_quantized_graph):
        report = estimate_fpga(xor_quantized_graph, ICE40_UP5K, "sequential")
        assert report.bram_bits_used > 0

    def test_report_str(self, xor_quantized_graph):
        report = estimate_fpga(xor_quantized_graph, ICE40_UP5K, "combinational")
        s = str(report)
        assert "FPGA Estimate" in s
        assert "iCE40UP5K" in s

    def test_ecp5_has_dsps(self, xor_quantized_graph):
        report = estimate_fpga(xor_quantized_graph, ECP5_25K, "sequential")
        assert report.dsp_slices_used >= 0

    def test_xor_fits_on_ice40(self, xor_quantized_graph):
        report = estimate_fpga(xor_quantized_graph, ICE40_UP5K, "sequential")
        assert report.fits

    def test_devices_dict(self):
        assert "ice40up5k" in DEVICES
        assert "ecp5-25k" in DEVICES


class TestBuildScript:
    def test_generates_makefile(self, xor_quantized_graph, output_dir):
        path = generate_build_script(
            xor_quantized_graph, ICE40_UP5K, output_dir)
        assert path.endswith("Makefile")
        with open(path) as f:
            content = f.read()
        assert "yosys" in content
        assert "nextpnr" in content

    def test_ecp5_makefile(self, xor_quantized_graph, output_dir):
        path = generate_build_script(
            xor_quantized_graph, ECP5_25K, output_dir)
        with open(path) as f:
            content = f.read()
        assert "synth_ecp5" in content


class TestConstraints:
    def test_generates_pcf(self, xor_quantized_graph, output_dir):
        path = generate_constraints(
            xor_quantized_graph, ICE40_UP5K, output_dir)
        assert path.endswith(".pcf")

    def test_generates_lpf_for_ecp5(self, xor_quantized_graph, output_dir):
        path = generate_constraints(
            xor_quantized_graph, ECP5_25K, output_dir)
        assert path.endswith(".lpf")
