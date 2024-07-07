import tkinter as tk
from tkinter import ttk, filedialog
import reinforcement_main
import threading

root = tk.Tk()
root.title("Osu!mania Training")
agent=None
def start_training():
    reinforcement_main.training_running = True
    num_episodes = int(episodes_entry.get())
    training_thread = threading.Thread(target=reinforcement_main.train_agent, args=(agent, num_episodes,))
    training_thread.start()

def stop_training():
    reinforcement_main.training_running = False

def load_model():
    global agent
    file_path = filedialog.askopenfilename(initialdir="Osu_ai/")
    if file_path:
        agent = reinforcement_main.OsuManiaAgent()  
        agent.load_model(file_path)  

def play():
    play_thread = threading.Thread(target=play_game)
    play_thread.start()

def play_game():
    env = reinforcement_main.OsuManiaEnv()
    state = env.reset()
    done = False
    counter=0
    tmp=[]
    while not done:
        action = agent.select_action(state)
        tmp.append(action)
        state, done = env.step(action)
        counter+=1
    print(counter)
    print(tmp)
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
