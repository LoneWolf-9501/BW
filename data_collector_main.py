import serial
from datetime import datetime, timedelta
from time import sleep
import csv
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger_tool import setup_logger

LOG = setup_logger('data_collector')

TIME_STRING_FORMAT = "%Y-%m-%d-%H-%M-%S.%f"
DATA_FOLDER = os.path.join(os.getcwd(), "init_tests/ODR_check/new_25")
# CSV_HEADER = ['index', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z', 'tag', 'datetime', 'activity']
# CSV_HEADER = ['index', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z', 'datetime', 'activity']
CSV_HEADER = ['index', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'datetime']

DEFAULT_LABEL = 'farm_ft'
DATA_COLLECTION_INTERVAL = timedelta(minutes=5)

def serial_init(ports, baudrate=115200, timeout=0.1) -> serial.Serial:

    for port in ports:

        try:
            hw_serial = serial.Serial(
                port=port,
                baudrate=baudrate,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=timeout,
            )

            LOG.info(f"Successfully connected to {port}")
            return hw_serial

        except (serial.SerialException, FileNotFoundError) as se:
            LOG.info(f"Failed to connect to {port}: {se}")
        
    raise Exception("All specified ports failed to connect.")

def get_current_dt(str_flag=False):
    return datetime.now().strftime(TIME_STRING_FORMAT) if str_flag else datetime.now()

def gen_file_path(act: str) -> str:
    os.makedirs(DATA_FOLDER, exist_ok=True)
    return f"{DATA_FOLDER}/{act}_{get_current_dt(str_flag=True)}.csv"

def write_to_csv(file_path, data):

    if len(data) == 0:
        LOG.warning("Data is empty. Skipping writing to csv.")
        return

    with open(file_path, "a+", newline="") as current_file:

        csv_writer = csv.writer(current_file)

        if os.stat(file_path).st_size == 0:
            csv_writer.writerow(CSV_HEADER)

        csv_writer.writerows(data)

def data_collector(ser_port: serial.Serial, label: str = None):

    """
    Collects data from the serial port and writes it to a CSV file.

    Args:
        ser_port (serial.Serial): The serial port to read data from.
        label (str, optional): An label to append to the data during live data collection.

    """
    
    data_buffer = []
    start_time = get_current_dt()

    while True:
        try:
            s_data = ser_port.readline()
            
            if s_data:
                print(f's_data = {s_data}')
                utf_data = s_data.decode("utf-8").strip().strip('\x00').strip('**')

                data_list = utf_data.split(",")  # Separate data using comma
                data_list.append(get_current_dt(str_flag=True))  # Append timestamp
                data_list.append(label)

                if len(data_list) == len(CSV_HEADER):
                    data_buffer.append(data_list)
                    
                else:
                    LOG.error(f"Length Mismatch. DATA Received - {len(data_list)}, Data Length specified -  {len(CSV_HEADER)}\nDATA Received - {data_list}\n\n")

                if label is None and (get_current_dt() - start_time >= DATA_COLLECTION_INTERVAL):

                        start_time = get_current_dt()
                        file_path = gen_file_path(DEFAULT_LABEL)

                        write_to_csv(file_path, data_buffer)  # Write to CSV

                        LOG.info(f'File generation complete -> : {file_path}')
                        data_buffer.clear()  # Clear the buffer after writing
            
            else:
                LOG.error('No s_data recieved')

        except KeyboardInterrupt:
            
            file_path = gen_file_path(label if label else DEFAULT_LABEL) 

            write_to_csv(file_path, data_buffer)  # Write to CSV

            if os.path.exists(file_path):
                LOG.info(f'File generation complete -> : {file_path}')

            print('Activity Ended')
            sleep(1)

            main()
        
        except Exception as e:
            LOG.error(f"Error {e}")


ports_to_try = ['/dev/ttyUSB0', '/dev/ttyUSB1']
hw_serial = serial_init(ports_to_try, baudrate=19200)

def main():

    os.system('clear')

    print('\nData Collector')
    print('\t1. Live')
    print('\t2. Time Interval')
    print('\t3. Exit')

    ch = int(input('Enter your choice: ').strip())

    if ch == 1:
        ip = input("ENTER ACTIVITY: ").split(':')[-1]
        data_collector(ser_port=hw_serial, label=ip)

    elif ch == 2:
        data_collector(ser_port=hw_serial)

    elif ch == 3:
        print("\nExiting...")
        sys.exit(0)

    else:
        print('Invalid Choice')

if __name__ == '__main__':
    main()