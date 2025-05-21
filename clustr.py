import numpy as np
from collections import defaultdict
import math
from typing import List, Tuple, Union
import struct

# ====================== BITSTREAM I/O ======================
class BitStream:
    def __init__(self):
        self.buffer = bytearray()
        self.bit_pos = 0
        self.current_byte = 0

    def write_bits(self, value: int, num_bits: int):
        for _ in range(num_bits):
            self.current_byte |= ((value >> (num_bits - 1)) & 1) << (7 - self.bit_pos)
            value <<= 1
            self.bit_pos += 1
            if self.bit_pos == 8:
                self.buffer.append(self.current_byte)
                self.current_byte = 0
                self.bit_pos = 0

    def flush(self):
        if self.bit_pos > 0:
            self.buffer.append(self.current_byte)

# ====================== tANS IMPLEMENTATION ======================
class tANS:
    def __init__(self, symbol_freq: dict, table_bits=12):
        self.TABLE_SIZE = 1 << table_bits
        self.STATE_BITS = 16
        self.MAX_STATE = (1 << self.STATE_BITS) - 1
        
        # Normalize frequencies
        total_freq = sum(symbol_freq.values())
        self.symbols = list(symbol_freq.keys())
        self.freqs = np.array([symbol_freq[s] for s in self.symbols], dtype=np.uint32)
        self.cum_freq = np.cumsum([0] + self.freqs.tolist()[:-1])
        
        # Build encoding/decoding tables
        self.encode_table = {}
        self.decode_table = np.zeros((self.TABLE_SIZE, 2), dtype=np.uint32)
        
        slot = 0
        for i, sym in enumerate(self.symbols):
            for j in range(self.freqs[i]):
                x = self.TABLE_SIZE // self.freqs[i]
                self.encode_table[(sym, j)] = (x << 16) | self.freqs[i]
                self.decode_table[slot] = [i, j]  # Store symbol index and offset
                slot += 1

    def encode(self, symbol: Union[str, Tuple], bitstream: BitStream, state: int) -> int:
        symbol_idx = self.symbols.index(symbol)
        freq = self.freqs[symbol_idx]
        offset = self.cum_freq[symbol_idx]
        
        # Output least significant bits
        bits_to_output = state % freq
        bitstream.write_bits(bits_to_output, int(math.log2(freq)))
        
        # Update state
        return (state // freq) * self.TABLE_SIZE + offset

    def decode(self, bitstream: BitStream, state: int) -> Tuple[Union[str, Tuple], int]:
        slot = state % self.TABLE_SIZE
        symbol_idx, offset = self.decode_table[slot]
        freq = self.freqs[symbol_idx]
        
        # Read bits
        bits_needed = int(math.log2(freq))
        bits = bitstream.read_bits(bits_needed)  # Implement read_bits similarly
        
        # Update state
        new_state = (state // self.TABLE_SIZE) * freq + offset + bits
        return self.symbols[symbol_idx], new_state

# ====================== CLUSTR CORE ======================
class ClustrCompressor:
    ESCAPE_SYMBOL = "##ESCAPE##"
    
    def __init__(self, max_short_run=32):
        self.max_short_run = max_short_run
        self.dynamic_threshold = max_short_run // 2
        self.bitstream = BitStream()
        self.symbol_freq = defaultdict(int)
        self.run_freq = defaultdict(int)
        self.total_symbols = 0
        
        # For neural predictor (conceptual)
        self.run_history = []
        self.nn_threshold_predictor = self._init_neural_predictor()

    def _init_neural_predictor(self):
        """Conceptual tiny neural net for dynamic threshold prediction"""
        # In practice: Use TensorFlow Lite or ONNX runtime
        return lambda x: max(4, min(self.max_short_run, int(np.mean(x) * 1.5)))

    def _simd_like_run_detect(self, data: bytes, pos: int) -> int:
        """Emulate SIMD run detection in Python"""
        if pos >= len(data):
            return 0
            
        symbol = data[pos]
        max_pos = min(pos + 256, len(data))  # Scan up to 256 bytes ahead
        
        # Emulate SIMD by checking 8 bytes at a time
        run_length = 0
        while pos + run_length < max_pos and data[pos + run_length] == symbol:
            run_length += 1
            # Early exit if we detect a non-match every 8 bytes
            if run_length % 8 == 0 and data[pos + run_length-1] != symbol:
                run_length -= 1
                break
                
        return run_length

    def _update_dynamic_threshold(self, observed_run: int):
        """Adjust thresholds based on recent runs"""
        self.run_history.append(observed_run)
        if len(self.run_history) > 100:
            self.run_history.pop(0)
        
        # Neural prediction or simple heuristic
        if len(self.run_history) > 10:
            self.dynamic_threshold = self.nn_threshold_predictor(self.run_history)

    def _encode_long_run(self, symbol: str, length: int, bitstream: BitStream):
        """Escape mechanism for long runs (L > 256)"""
        # Format: [ESCAPE_SYMBOL, symbol, varint(length)]
        bitstream.write_symbol(self.ESCAPE_SYMBOL)
        bitstream.write_symbol(symbol)
        
        # Varint encoding
        while length > 0x7F:
            bitstream.write_bits((length & 0x7F) | 0x80, 8)
            length >>= 7
        bitstream.write_bits(length, 8)

    def compress(self, data: bytes) -> bytes:
        """Full compression pipeline with tANS"""
        # First pass: Collect frequencies
        pos = 0
        while pos < len(data):
            run_length = self._simd_like_run_detect(data, pos)
            symbol = data[pos]
            
            if run_length > self.dynamic_threshold:
                quantized_L = min(run_length, 256)
                self.run_freq[(symbol, quantized_L)] += 1
                if run_length > 256:
                    self._encode_long_run(symbol, run_length, self.bitstream)
            else:
                self.symbol_freq[symbol] += 1
                
            self._update_dynamic_threshold(run_length)
            pos += run_length
        
        # Build tANS tables
        merged_freq = {**self.symbol_freq, **self.run_freq}
        if self.ESCAPE_SYMBOL in merged_freq:
            del merged_freq[self.ESCAPE_SYMBOL]  # Escape has fixed freq
        ans = tANS(merged_freq)
        
        # Second pass: Actual encoding
        state = ans.TABLE_SIZE
        pos = 0
        while pos < len(data):
            run_length = self._simd_like_run_detect(data, pos)
            symbol = data[pos]
            
            if run_length > self.dynamic_threshold:
                quantized_L = min(run_length, 256)
                state = ans.encode((symbol, quantized_L), self.bitstream, state)
                if run_length > 256:
                    self._encode_long_run(symbol, run_length, self.bitstream)
            else:
                for _ in range(run_length):
                    state = ans.encode(symbol, self.bitstream, state)
                    
            pos += run_length
        
        self.bitstream.flush()
        return bytes(self.bitstream.buffer)

# ====================== BENCHMARKING ======================
if __name__ == "__main__":
    # Test data
    data = b"AAAA" * 1000 + b"BB" * 500 + b"C" * 200 + b"XYZ" * 50
    
    # Compress
    clustr = ClustrCompressor(max_short_run=64)
    compressed = clustr.compress(data)
    
    print(f"Original: {len(data)} bytes")
    print(f"Compressed: {len(compressed)} bytes")
    print(f"Compression ratio: {len(compressed)/len(data):.2%}")
    
    # Compare with zlib
    import zlib
    zlib_compressed = zlib.compress(data)
    print(f"zlib: {len(zlib_compressed)} bytes ({len(zlib_compressed)/len(data):.2%})")