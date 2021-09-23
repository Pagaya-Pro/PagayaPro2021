import random
from typing import Iterable, List

from courses.planet_wars.planet_wars import Player, PlanetWars, Order, Planet
from courses.planet_wars.tournament import get_map_by_id, run_and_view_battle, TestBot

import pandas as pd
import math


class dont_be_mean_bot(Player):

    def distance(self,planet1,planet2):
        dx = (planet1.x-planet2.x)**2
        dy = (planet1.y - planet2.y) ** 2
        return math.ceil(math.sqrt(dx+dy))


    def all_distances(self,planet, gameWars):
        pass

    def play_turn(self, game: PlanetWars) -> Iterable[Order]:
        planet_df = game.get_planets_data_frame()
        fleet_df = game.get_fleets_data_frame()
        #planet_df.head()
        return []
# defense

#steal



