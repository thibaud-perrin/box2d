from multiprocessing import Process
import gymnasium as gym
import numpy as np

from PIL import Image
import imageio
import cv2

class Environment(Process):
    def __init__(self, env_idx, child_conn, env_name, visualize=False):
        super(Environment, self).__init__()
        self.env_name = env_name
        self.env = gym.make(self.env_name, render_mode="rgb_array")
        self.is_render = visualize
        self.env_idx = env_idx
        self.child_conn = child_conn
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]

    def run(self):
        super(Environment, self).run()
        iteration = 0

        # Worker infinit loop
        while True:
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            self.child_conn.send(state)
            
            # variables
            done = False
            timestep = 0
            
            # Iteration loop
            while not done:
                # Check if there's something to receive
                print(f"\rWorker=<{self.env_idx}> Iteration={iteration +1} Timestep={timestep +1}", end="")
                # waiting for the action
                action = self.child_conn.recv()

                # Printing env render (rgb_array)
                if self.is_render and self.env_idx == 0:
                    screen = self.env.render()
                    # Add title to the screen
                    screen = cv2.putText(
                        np.array(screen),
                        f"W<{self.env_idx}> I[{iteration +1}] T=[{timestep +1}]",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA
                    )
                    self.child_conn.send(screen)

                # Take action A and receive the next state S' and reward R
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                state = np.reshape(state, [1, self.state_size])
                # sending the env information (next_state, reward, done, info)
                self.child_conn.send([state, reward, done, info])

                timestep += 1
                
            message = self.child_conn.recv() # Only call recv() if there's a message
            if message == 'terminate':  # Check for termination message
                print(f"\nWorker=<{self.env_idx}> Done", end="\n")
                break
            iteration += 1
        
    def save_mp4(self, img_list):
        path = f"./img/{self.env_name}-{self.env_idx}.mp4"
        print(f"Saving {path}....")
        
        # Convert the list of frames to a numpy array
        resized_img_array = []
        for img in img_list:
            img_pil = Image.fromarray(img)
            # Make sure width and height are divisible by 16
            img_resized_pil = img_pil.resize((608, 400))
            img_resized = np.array(img_resized_pil)
            resized_img_array.append(img_resized)
        
        # Create mp4 video
        imageio.mimsave(path, resized_img_array, fps=20)
        
        return path
    
    def convert_mp4_to_gif(self, input_file_path, output_file_path):
        # Read video frames
        reader = imageio.get_reader(input_file_path)
        fps = reader.get_meta_data()['fps']
        frames = [frame for frame in reader]

        print(f"Saving {output_file_path}....")
        # Save as gif
        imageio.mimsave(output_file_path, frames, 'GIF', duration=int(1000 * 1/fps), loop=0)
        # writer = imageio.get_writer(output_file_path, duration=int(1000 * 1/fps))
        # for i, im in enumerate(reader):
        #     writer.append_data(im)
        # writer.close()
