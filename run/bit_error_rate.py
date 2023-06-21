import sys
import time
import random
from bitarray import bitarray

import larpix
import random
import power_up_v2

def main(*args, **kwargs):
    print('loopback config')

    # create controller
    #c = power_up_v2.main(logger=True, *args, **kwargs)
    c = power_up_v2.main()

    c.io.group_packets_by_io_group = True
    c.io.double_send_packets = False

    packet_duration = 3.2e-5 #1.6e-5 #3.2e-5 # seconds (packet length with 4 clk cycles per bit +20%)
    
    n_packets = 1
    iteration_duration = max(packet_duration * n_packets,0.1)
    iteration = 0
    sent_packets = 0
    missing = 0
    errors = 0

    with open('v2b_ber_test___errors.txt','w') as fe:
        fe.write('Bit error rate testv   ERRORS    November 1, 2021\n')

    with open('v2b_ber_test___log.txt','w') as fl:
        fl.write('Bit error rate testv   LOG    November 1, 2021\n')
        fl.write('iter\tsent\tmissing\terrors\n')
    
    print('running for',iteration_duration,'s, sending',n_packets,'packets on each iteration')    
    while True:
        iteration += 1

        test_packet = larpix.Packet_v2()
        test_packet.io_group = 1
        test_packet.io_channel = 4
        test_packet.channel_id = random.randint(0,63)
        test_packet.timestamp = random.randint(0,2147483647)
        test_packet.chip_id = random.randint(0,255)
        test_packet.packet_type = 1
        test_packet.downstream_marker = 1 #0 # upstream packet
        test_packet.assign_parity()
        #####print('INITIAL TEST PACKET: ',test_packet.export()['bits'])
        #print('INITIAL TEST PACKET: ',test_packet.bits)
        test_packets = [test_packet]*n_packets
    
        packets, bytestream = [], b''
        p,b = [], b''
        #c.logger.enable()
        c.start_listening()
        #time.sleep(0.1)
        time.sleep (0.001)
        c.send(test_packets)
        now = time.time()        
        while time.time() - now < iteration_duration:
            p,b = c.read()
            packets += p
            bytestream += b
        while p:
            # keep reading as long as there is data
            p,b = c.read()
            packets += p
            bytestream += b
        c.stop_listening()
        #c.logger.disable()
        c.store_packets(packets, bytestream, str(iteration))

        data_packets = larpix.PacketCollection([pkt for pkt in c.reads[-1] if isinstance(pkt,larpix.Packet_v2)]).extract('bits')
        #print('data packets: ',data_packets)
        #test_packet.chip_id = 1
        #test_packet.assign_parity()
        for bits in data_packets:
            if bits != test_packet.bits:
                print(test_packet.bits)
                print(bits)
                errors += 1
                with open('v2b_ber_test___errors.txt','a') as fe:
                    s = str(test_packet.bits)+'    '+str(bits)+'\n'
                    fe.writelines(s)
        
        missing = max(n_packets - len(data_packets),0)
        sent_packets += n_packets
        missing += missing

        #print('iteration {}\t\t sent packets {}\t\t missing packets {}\t\t packet errors {}'.format(
        #    iteration,
        #    sent_packets,
        #    missing,
        #    errors
        #), end='\r')

        if iteration % 1000 == 0:
            print('iteration {}\t\t sent packets {}\t\t missing packets {}\t\t packet errors {}'.format(
            iteration,
            sent_packets,
            missing,
            errors))
            #print()
            with open('v2b_ber_test___log.txt','a') as fl:
                s = str(iteration)+'\t'+str(sent_packets)+'\t'+str(missing)+'\t'+str(errors)+'\n'
                fl.writelines(s)

        c.reads = []
    
    return c

if __name__ == '__main__':
    c = main(*sys.argv[1:])
