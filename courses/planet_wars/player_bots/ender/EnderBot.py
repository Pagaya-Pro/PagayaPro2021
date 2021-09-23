from typing import Iterable

import pandas as pd

from courses.planet_wars.planet_wars import Player, PlanetWars, Order, Planet
import numpy as np


class EnderBot(Player):

    NAME = "Ender"

    # def new_game_has_started(self, game: PlanetWars):
    #     """
    #     This function will be called at the beginning of each game.
    #     Here is the place to restart the game state.
    #     for example if you count the number of ships you sent in fleets here is the place to set this counter back to 0
    #     Note: Exception here will make you lose the game
    #     :param game: PlanetWars object representing the map initial state
    #     """
    #     self._scores = pd.DataFrame(columns=['sum', 'growth', 'dist', 'ships', 'foreign'])

    def score(self, src: Planet, dest: Planet, game: PlanetWars, destinations: set):
        """
        Score function for sending a fleet from source to destination
        :param src:
        :param dest:
        :param game:
        :param destinations:
        :return:
        """
        def ship_dest_score():
            dest_ships_on_arrival = dest.num_ships + dist * dest.growth_rate
            # if (1 - dest_ships_on_arrival / src.num_ships) > 0:
            # print((src.num_ships - dest_ships_on_arrival) / max(src.num_ships, dest_ships_on_arrival, 1))
            return 1 - dest_ships_on_arrival / src.num_ships
            # return 10 * (src.num_ships - dest_ships_on_arrival) / max(src.num_ships, dest_ships_on_arrival, 1)

        dist = Planet.distance_between_planets(src, dest)
        turns_left = 200 - game.turns

        # if turns_left < dist or dest.planet_id in destinations or src.num_ships == 0:
        if dest.planet_id in destinations or src.num_ships == 0:
            # if src.num_ships == 0:
            return -np.inf

        dist_score = 4 / dist
        foreign_score = 1 if dest.owner == game.NEUTRAL else 2
        growth_rate_score = 3 * (dest.growth_rate - src.growth_rate) / max(src.growth_rate, dest.growth_rate, 1)
        ship_score = ship_dest_score()

        final_score = foreign_score + (dist_score + growth_rate_score) + ship_score

        # self._scores = self._scores.append({'sum': final_score,
        #                                     'growth': growth_rate_score,
        #                                     'dist': dist_score,
        #                                     'ships':ship_score,
        #                                     'foreign': foreign_score}, ignore_index=True)
        # print(final_score)
        return final_score

    def ships_to_send_in_a_flee(self, source_planet: Planet, dest_planet: Planet) -> int:
        dest_ships_on_arrival = dest_planet.num_ships + \
                                Planet.distance_between_planets(source_planet, dest_planet) * dest_planet.growth_rate
        return max(dest_ships_on_arrival + 1, int(source_planet.num_ships * 0.75))

    def play_turn(self, game: PlanetWars) -> Iterable[Order]:
        """
        See player.play_turn documentation.
        :param game: PlanetWars object representing the map - use it to fetch all the planets and flees in the map.
        :return: List of orders to execute, each order sends ship from a planet I own to other planet.
        """
        orders = []
        try:
            # dests = {fleet.destination_planet_id for fleet in game.get_fleets_by_owner(game.ME)}
            dests = set()

            my_planets = game.get_planets_by_owner(game.ME)
            foreign_planets = game.get_planets_by_owner(game.ENEMY) + game.get_planets_by_owner(game.NEUTRAL)

            for planet in my_planets:
                try:
                    targets = [(dest, self.score(planet, dest, game, dests)) for dest in foreign_planets]
                    best_target = max(targets, key=lambda target: target[1])
                    # print(best_target[1])
                    if best_target[1] > -np.inf:
                        fleet_size = self.ships_to_send_in_a_flee(planet, best_target[0])
                        orders.append(Order(planet.planet_id, best_target[0].planet_id, fleet_size))
                        dests.add(best_target[0])

                except:
                    continue

        except:
            pass

        finally:
            # print(self._scores)
            # self._scores = pd.DataFrame(columns=['sum', 'growth', 'dist', 'ships', 'foreign'])
            return orders

