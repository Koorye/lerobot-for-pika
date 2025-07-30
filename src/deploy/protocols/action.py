from dataclasses import dataclass


@dataclass
class Action:
    robot_name: str
    actions: list[float]

    def to_dict(self):
        return {
            'robot_name': self.robot_name,
            'actions': self.actions,
        }
    
    @staticmethod
    def from_dict(data):
        return Action(
            robot_name=data['robot_name'],
            actions=data['actions'],
        )
