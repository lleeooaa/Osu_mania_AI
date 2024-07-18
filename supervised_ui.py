import tkinter as tk
from tkinter import ttk, filedialog
import supervised_main
import threading
import time

root = tk.Tk()
root.title("Osu!mania Training")
agent=None
def start_training():
    supervised_main.training_running = True
    num_episodes = int(episodes_entry.get())
    training_thread = threading.Thread(target=supervised_main.train_agent, args=(agent, num_episodes,))
    training_thread.start()

def stop_training():
    supervised_main.training_running = False

def load_model():
    global agent
    file_path = filedialog.askopenfilename(initialdir="Osu_ai/")
    if file_path:
        agent = supervised_main.OsuManiaAgent()  
        agent.load_model(file_path) 

def play():
    play_thread = threading.Thread(target=play_game)
    play_thread.start()

def play_game():
    env = supervised_main.OsuManiaEnv()
    state = env.reset()
    done = False
    counter=0
    while not done:
        #time.sleep(0.003)
        action = agent.select_action(state)
        env.perform_action(action)
        state, _, done = env.step()
        counter+=1
    print(counter)

episodes_label = ttk.Label(root, text="Number of Episodes:")
episodes_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

episodes_entry = ttk.Entry(root)
episodes_entry.grid(row=0, column=1, padx=5, pady=5)

start_button = ttk.Button(root, text="Start Training", command=start_training)
start_button.grid(row=1, column=0, padx=5, pady=5)

stop_button = ttk.Button(root, text="Stop Training", command=stop_training)
stop_button.grid(row=1, column=1, padx=5, pady=5)

load_button = ttk.Button(root, text="Load Model", command=load_model)
load_button.grid(row=2, column=0, padx=5, pady=5)

play_button = ttk.Button(root, text="Play", command=play)
play_button.grid(row=2, column=1, padx=5, pady=5)


root.mainloop()
