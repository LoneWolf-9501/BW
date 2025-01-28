import matplotlib
matplotlib.use('TkAgg')  # Use an interactive backend

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils.logger_tool import setup_logger

import numpy as np
from datetime import datetime
import serial
import sys

LOG = setup_logger('data_visualiser')


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
        

# Create figure for plotting
fig, axs = plt.subplots(3, 3, figsize=(15, 10))

lines = []

# Circular buffer for acc_x
buffer_size = 25
data_buffer = np.zeros((buffer_size, 9), dtype=float)  # Preallocate buffer
index = 0  # Current index in the circular buffer

axes = ['x', 'y', 'z']
# sensors = ['acc', 'gyro']
sensors = ['acc', 'gyro', 'mag']

for i, sensor in enumerate(sensors):
    for j, axis in enumerate(axes):

        ax = axs[i, j]

        ax.set_title(f'{sensor}_{axis}', pad=15)
        
        ax.set_ylabel(f'{sensor}_{axis} values', labelpad=15)

        ax.set_xlim(0, buffer_size)

        ax.set_ylim(-250 if sensor == 'mag' else -20, 250 if sensor == 'mag' else 20)

        line, = ax.plot([], [], lw=1)
        lines.append(line)

def update(frame):
    global index, data_buffer, buffer_size
    try:
        s_data = hw_serial.readline()
        
        if s_data:
            utf_data = s_data.decode("utf-8").strip().strip('\x00').strip('**')
            data = utf_data.split(",")

            if len(data) != 10:
                raise ValueError(f'Invalid data length - {data} - {len(data)}')
            
            print(data)

            # Update data buffers
            data_buffer[index, 0] = (float(data[1]))      # acc_x
            data_buffer[index, 1] = (float(data[2]))      # acc_y
            data_buffer[index, 2] = (float(data[3]))      # acc_z
            data_buffer[index, 3] = (float(data[4]))      # gyro_x
            data_buffer[index, 4] = (float(data[5]))      # gyro_y
            data_buffer[index, 5] = (float(data[6]))      # gyro_z
            data_buffer[index, 6] = (float(data[7]))      # mag_x
            data_buffer[index, 7] = (float(data[8]))      # mag_y
            data_buffer[index, 8] = (float(data[9]))      # mag_z

            # Update data buffer in a circular manner
            index = (index + 1) % buffer_size  # Wrap around

            current_time = datetime.now()
            
            for i, line in enumerate(lines):
                line.set_data(range(buffer_size), data_buffer[:, i])

        else:
            LOG.error('No s_data')

    except Exception as e:
        LOG.error(f"Error: {e}")
        print("Error:", e)
        

    return lines

def frame_gen():
    while True:
        yield 0

try:
    
    ports_to_try = ['/dev/ttyUSB0', '/dev/ttyUSB1']
    hw_serial = serial_init(ports_to_try, baudrate=19200)
    
    # Create an animation
    ani = animation.FuncAnimation(fig, update, frames=frame_gen, blit=True, interval=5, save_count=50)


    plt.tight_layout()
    plt.show()

except KeyboardInterrupt:
    print("Exiting...")    
    LOG.info("Exiting...")

except Exception as e:
    print("Error:", e)    
    LOG.error("Error:", e)

finally:
    hw_serial.close()  # Ensure the serial port is closed
    plt.close()
    sys.exit(0)  # Exit the program