"""
Tests for w2s.emit — Verilog code-emission helpers.
"""

import pytest

from w2s.emit import acc_bits_for, sign_extend_expr, slit, ulit


# ---------------------------------------------------------------------------
#  slit() — signed Verilog literal
# ---------------------------------------------------------------------------

class TestSlit:
    def test_positive(self):
        assert slit(8, 42) == "8'sd42"

    def test_zero(self):
        assert slit(8, 0) == "8'sd0"

    def test_negative_parenthesized(self):
        result = slit(8, -5)
        assert result == "(-8'sd5)"

    def test_negative_is_parenthesized(self):
        """Negative literals must be wrapped in parens for Verilog correctness."""
        result = slit(16, -100)
        assert result.startswith("(")
        assert result.endswith(")")

    def test_large_value(self):
        result = slit(32, 2**30)
        assert "32'sd" in result
        assert str(2**30) in result

    def test_different_bit_widths(self):
        assert "4'sd" in slit(4, 7)
        assert "16'sd" in slit(16, 127)
        assert "32'sd" in slit(32, 0)


# ---------------------------------------------------------------------------
#  ulit() — unsigned Verilog literal
# ---------------------------------------------------------------------------

class TestUlit:
    def test_positive(self):
        assert ulit(8, 42) == "8'd42"

    def test_zero(self):
        assert ulit(8, 0) == "8'd0"

    def test_large_value(self):
        result = ulit(16, 65535)
        assert result == "16'd65535"


# ---------------------------------------------------------------------------
#  sign_extend_expr()
# ---------------------------------------------------------------------------

class TestSignExtendExpr:
    def test_basic_extension(self):
        """sign_extend_expr(foo, 8, 32) should produce a Verilog replication."""
        result = sign_extend_expr("foo", 8, 32)
        # Should reference foo[7] for the sign bit
        assert "foo[7]" in result
        # Should have the replication count of 24
        assert "24" in result

    def test_output_contains_wire_name(self):
        result = sign_extend_expr("my_wire", 4, 16)
        assert "my_wire" in result

    def test_sign_bit_index(self):
        """For 4-bit to 16-bit, the sign bit should be bit [3]."""
        result = sign_extend_expr("w", 4, 16)
        assert "w[3]" in result


# ---------------------------------------------------------------------------
#  acc_bits_for()
# ---------------------------------------------------------------------------

class TestAccBitsFor:
    def test_small_input(self):
        """For 2 inputs at 8 bits, accumulator should be > 16."""
        result = acc_bits_for(2, 8)
        assert result > 16

    def test_larger_input(self):
        """More inputs should require more accumulator bits."""
        small = acc_bits_for(4, 8)
        large = acc_bits_for(1024, 8)
        assert large > small

    def test_wider_weights_need_more_bits(self):
        """16-bit weights should need more accumulator bits than 8-bit."""
        result_8 = acc_bits_for(10, 8)
        result_16 = acc_bits_for(10, 16)
        assert result_16 > result_8

    def test_minimum_value(self):
        """Even for 2 inputs, accumulator should be reasonably sized."""
        result = acc_bits_for(2, 4)
        assert result >= 10  # 2*4 + 1 + 2 = 11 minimum
