import time


class Client(object):
    def __init__(
        self,
        host,
        port,
        camera, 
        robot, 
        prompt,
        frequency,
        visualize_every,
    ):
        self.host = host
        self.port = port
        self.prompt = prompt
        self.camera = camera
        self.robot = robot
        self.frequency = frequency
        self.visualize_every = visualize_every

        self.robot.reset()
        self.count = 0

        self.policy = WebsocketClientPolicy(host, port)

    def run(self):
        while True:
            self.do_action()
            self.count += 1
            if self.visualize_every > 0 and self.count % self.visualize_every == 0:
                self.robot.visualize()
            time.sleep(1 / self.frequency)
    
    def do_action(self):
        obs = self.camera.get_observation()
        obs.update(self.robot.get_observation())
        obs['prompt'] = self.prompt
        
        # left_rgb = obs['left_wrist_fisheye_rgb']
        # right_rgb = obs['right_wrist_fisheye_rgb']
        # rgb = np.concatenate([left_rgb, right_rgb], axis=1)
        # imageio.imwrite('test.jpg', rgb)
        actions_list = self.policy.infer(obs)['actions'][:4]

        for actions in actions_list:
            self.robot.do_action(actions)
