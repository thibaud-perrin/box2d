from multiprocessing import Process
import gymnasium as gym
import numpy as np

from PIL import Image
import imageio
import cv2
import time

class Environment(Process):
    def __init__(self, env_idx, child_conn, env_name, current_episode = 0, visualize=False, print_episode = 50):
        super(Environment, self).__init__()
        self.env_name = env_name
        self.env = gym.make(self.env_name, render_mode="rgb_array")
        self.is_render = visualize
        self.env_idx = env_idx
        self.child_conn = child_conn
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.current_episode = current_episode
        self.print_episode = print_episode
        
    def set_current_episode(self, current_episode=0):
        self.current_episode = current_episode
        
    def run(self):
        super(Environment, self).run()
        iteration = 0
        msg_title = ""
        msg_info = None

        # Worker infinit loop
        while True:
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            # 1 - Sending initial state
            msg_title = "1 - INITIAL_STATE"
            msg_info = state
            self.child_conn.send([msg_title, msg_info])
            
            # variables
            done = False
            timestep = 0
            screen_list = []
            
            # 2- Iteration loop
            while not done:
                # 3 - Sending loop infos
                msg_title = "3 - LOOP_INFOS"
                msg_info = [self.env_idx, iteration +1, self.current_episode, timestep +1]
                self.child_conn.send([msg_title, msg_info])
                # print(f"\rWorker=<{self.env_idx}> Iteration={iteration +1} Timestep={timestep +1}", end="")
                # 4 - waiting for the action
                action = self.child_conn.recv()

                # Printing env render (rgb_array)
                if self.is_render and self.current_episode % self.print_episode == 0:
                    screen = self.env.render()
                    # Add title to the screen
                    screen = cv2.putText(
                        np.array(screen),
                        f"W<{self.env_idx}> I[{iteration +1}] T=[{timestep +1}]",
                        (25, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA
                    )
                    screen_list.append(screen)

                # Take action A and receive the next state S' and reward R
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                state = np.reshape(state, [1, self.state_size])
                # 5 - Sending the env information (next_state, reward, done, info)
                msg_title = "5 - ENV_STEP"
                msg_info = [state, reward, done, info]
                self.child_conn.send([msg_title, msg_info])

                timestep += 1
            
            # if we want to save the image
            if self.is_render and self.current_episode % self.print_episode == 0:
                path = f"./img/{self.env_name}-w{self.env_idx}-e{self.current_episode}.gif"
                self.save_gif(screen_list, path)
            
            # 8 - Sending Iteration end
            msg_title = "8 - ITERATION_DONE"
            msg_info = True
            self.child_conn.send([msg_title, msg_info])
            
            # 9 Checking if Recieved terminate message
            message = self.child_conn.recv() # Only call recv() if there's a message
            if message == 'terminate':  # Check for termination message
                print("=============================")
                print("=============================")
                print(f"Worker=<{self.env_idx}> Done")
                print(" ////////////////////////////")
                print("////////////////////////////")
                break
            iteration += 1
        
    def save_gif(self, img_list, path):
        # 6 - Sending Saving game message
        msg_title = "6 - SAVING_GAME"
        msg_info = "Saving...."
        self.child_conn.send([msg_title, msg_info])
        time.sleep(0.025)
        
        # Convert the list of frames to a numpy array
        resized_img_array = []
        for img in img_list:
            img_pil = Image.fromarray(img)
            # Make sure width and height are divisible by 16
            img_resized_pil = img_pil.resize((608, 400))
            img_resized = np.array(img_resized_pil)
            resized_img_array.append(img_resized)
        
        # Create gif video
        fps = 20
        imageio.mimsave(path, resized_img_array, 'GIF', duration=int(1000 * 1/fps), loop=0)
        # 7 - Sending Saved game message
        msg_title = "7 - GAME_SAVED"
        msg_info = f"Saved in {path}"
        self.child_conn.send([msg_title, msg_info])
        time.sleep(0.025)