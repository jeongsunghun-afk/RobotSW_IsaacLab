import socket
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
import threading
import sys

# 설정
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
BUFFER_SIZE = 65535 # Increased buffer size for larger payload
HISTORY_LEN = 200

# 데이터 저장소
# joint_data[joint_name] = { "ref": deque, "pos": deque, "vel": deque, "trq": deque, "time": deque }
joint_data = {}
joint_names = []
current_joint_idx = 0
time_counter = 0

# 데이터 락 (쓰레드 간 동기화)
data_lock = threading.Lock()

# UDP 소켓 설정
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

print(f"Listening on {UDP_IP}:{UDP_PORT}...")
print("Press LEFT/RIGHT arrow keys to switch joints.")

def receive_data():
    global time_counter, joint_names
    while True:
        try:
            data, addr = sock.recvfrom(BUFFER_SIZE)
            parsed = json.loads(data.decode())
            
            # parsed: {'t': int, 'names': [str], 'ref': [float], 'pos': [float], 'vel': [float], 'trq': [float]}
            
            with data_lock:
                names = parsed['names']
                
                # Update joint names list if changed or first time
                if len(joint_names) != len(names):
                    joint_names = names
                    # Initialize deques for new joints
                    for name in names:
                        if name not in joint_data:
                            joint_data[name] = {
                                "ref": deque(maxlen=HISTORY_LEN),
                                "pos": deque(maxlen=HISTORY_LEN),
                                "vel": deque(maxlen=HISTORY_LEN),
                                "trq": deque(maxlen=HISTORY_LEN),
                                "time": deque(maxlen=HISTORY_LEN)
                            }
                
                # Store data
                if "t" in parsed:
                     t_step = parsed["t"]
                else:
                     t_step = time_counter
                     time_counter += 1
                
                for i, name in enumerate(names):
                    if name in joint_data:
                        joint_data[name]["ref"].append(parsed["ref"][i])
                        joint_data[name]["pos"].append(parsed["pos"][i])
                        joint_data[name]["vel"].append(parsed["vel"][i])
                        joint_data[name]["trq"].append(parsed["trq"][i])
                        joint_data[name]["time"].append(t_step)
                        
        except BlockingIOError:
            pass
        except Exception as e:
            # print(f"Error receiving: {e}")
            pass

# 수신 쓰레드 시작
thread = threading.Thread(target=receive_data, daemon=True)
thread.start()

# 그래프 초기화
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
lines = {}

# Action vs Position
line_ref, = axs[0].plot([], [], 'r--', label='Target (Action)')
line_pos, = axs[0].plot([], [], 'b-', label='Current (Position)')
axs[0].set_ylabel('Position [rad]')
axs[0].legend(loc='upper right')

# Position Error
line_err, = axs[1].plot([], [], 'g-', label='Error')
axs[1].set_ylabel('Error [rad]')
axs[1].grid(True)

# Velocity
line_vel, = axs[2].plot([], [], 'm-', label='Velocity')
axs[2].set_ylabel('Velocity [rad/s]')

# Torque
line_trq, = axs[3].plot([], [], 'k-', label='Torque')
axs[3].set_ylabel('Torque [Nm]')
axs[3].set_xlabel('Time Step')

def on_key(event):
    global current_joint_idx
    # print(f"Key pressed: {event.key}") # Debugging
    
    if not joint_names:
        print("Warning: No joint data received yet.")
        return
        
    if event.key == 'right':
        current_joint_idx = (current_joint_idx + 1) % len(joint_names)
        print(f"Switched to Next Joint: {joint_names[current_joint_idx]} ({current_joint_idx})")
    elif event.key == 'left':
        current_joint_idx = (current_joint_idx - 1 + len(joint_names)) % len(joint_names)
        print(f"Switched to Prev Joint: {joint_names[current_joint_idx]} ({current_joint_idx})")
    
    # Title update is done in update() function or here? 
    # Better in update() because it runs periodically.
    
fig.canvas.mpl_connect('key_press_event', on_key)

def update(frame):
    if not joint_names:
        axs[0].set_title("Waiting for data...")
        return
    
    # Safe access
    with data_lock:
        try:
            name = joint_names[current_joint_idx]
            data = joint_data[name]
            
            t = list(data["time"])
            if not t:
                return

            ref = list(data["ref"])
            pos = list(data["pos"])
            vel = list(data["vel"])
            trq = list(data["trq"])
            
            line_ref.set_data(t, ref)
            line_pos.set_data(t, pos)
            
            err = np.array(ref) - np.array(pos)
            line_err.set_data(t, err)
            
            line_vel.set_data(t, vel)
            line_trq.set_data(t, trq)
            
            axs[0].set_title(f"Joint [{current_joint_idx}/{len(joint_names)-1}]: {name}")
            
            for ax in axs:
                ax.relim()
                ax.autoscale_view()
                
        except Exception as e:
            print(f"Update error: {e}")

ani = animation.FuncAnimation(fig, update, interval=100) # 100ms update rate
plt.tight_layout()
# Window Positioning for TigerVNC
try:
    mngr = plt.get_current_fig_manager()
    # screen_width = mngr.window.winfo_screenwidth() # Tkinter specific
    # screen_height = mngr.window.winfo_screenheight()
    # Set to top-right corner. Adjust these values if needed.
    # Format: "widthxheight+x+y" or "+x+y"
    mngr.window.wm_geometry("+1500+100") 
except Exception as e:
    print(f"Failed to set window geometry: {e}")

plt.show()
