import os
import random
import dpkt

class PcapGenerator:
    """
    Class to generate PCAP files for testing purposes.
    """

    def __init__(self, filename):
        """
        Constructor for the PcapGenerator class.

        :param filename: The name of the PCAP file to be created.
        :raises ValueError: If the filename is not provided.
        """
        if not filename:
            raise ValueError("Filename must be provided.")
        self.filename = filename

    def generate_pcap(self, packet_count):
        """
        Generates a sample PCAP file with dummy packets.

        :param packet_count: The number of packets to generate.
        :raises ValueError: If packet_count is not a positive integer.
        """
        if not isinstance(packet_count, int) or packet_count <= 0:
            raise ValueError("Packet count must be a positive integer.")

        with open(self.filename, 'wb') as f:
            writer = dpkt.pcap.Writer(f)
            for i in range(packet_count):
                packet = self.create_dummy_packet(i)
                ts = i  # Timestamp (can be modified as needed)
                writer.writepkt(packet, ts)

        print(f"PCAP file '{self.filename}' generated with {packet_count} packets.")

    def create_dummy_packet(self, index):
        """
        Creates a dummy packet for testing.

        :param index: The index of the packet being created.
        :return: A bytes object containing the dummy packet data.
        """
        packet_data = bytearray(64)  # Create a buffer of 64 bytes
        packet_data[0:4] = index.to_bytes(4, 'little')  # Write the index at the start of the packet

        # Fill the rest of the packet with random data
        for i in range(4, len(packet_data)):
            packet_data[i] = random.randint(0, 255)

        # Assuming an Ethernet frame; you can adjust the frame structure to your needs
        eth = dpkt.ethernet.Ethernet(dst=b'\xff' * 6, src=b'\x00' * 6, type=dpkt.ethernet.ETH_TYPE_IP)
        eth.data = bytes(packet_data)

        return bytes(eth)

# Usage Example for PcapGenerator

# Example 1: Generating a PCAP file for network testing
try:
    pcap_generator = PcapGenerator('test.pcap')
    pcap_generator.generate_pcap(10)  # Generate a PCAP file with 10 dummy packets
except ValueError as error:
    print(f"Error generating PCAP file: {error}")

# Example 2: Generating a larger PCAP file for performance testing
try:
    large_pcap_generator = PcapGenerator('large_test.pcap')
    large_pcap_generator.generate_pcap(1000)  # Generate a PCAP file with 1000 dummy packets
except ValueError as error:
    print(f"Error generating large PCAP file: {error}")

# Example 3: Generating a PCAP file for packet capture
try:
    capture_pcap_generator = PcapGenerator('capture.pcap')
    capture_pcap_generator.generate_pcap(100)  # Generate a PCAP file with 100 dummy packets
except ValueError as error:
    print(f"Error generating PCAP file for packet capture: {error}")   

# Example 4: Generating a PCAP file for packet capture with a specific interface
try:
    capture_pcap_generator = PcapGenerator('capture.pcap')
    capture_pcap_generator.generate_pcap(100, interface='eth0')  # Generate a PCAP file with 100 dummy packets on the specified interface
except ValueError as error;

