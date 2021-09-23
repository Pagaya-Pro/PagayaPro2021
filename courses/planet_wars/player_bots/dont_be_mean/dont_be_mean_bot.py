from courses.planet_wars.planet_wars import Player
import math

class dont_be_mean_bot(Player):
    def distance(self,planet1,planet2):
        dx = (planet1.x-planet2.x)**2
        dy = (planet1.y - planet2.y) ** 2
        return math.ceil(math.sqrt(dx+dy))
