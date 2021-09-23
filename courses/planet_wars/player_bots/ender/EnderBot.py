from typing import Iterable

from courses.planet_wars.planet_wars import Player, PlanetWars, Order, Planet
import numpy as np


class EnderBot(Player):

    NAME = "Ender"

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
            dest_ships_on_arrival = dest.num_ships + Planet.distance_between_planets(src, dest) * dest.growth_rate
            return 1 - dest_ships_on_arrival / src.num_ships

        if dest.planet_id in destinations or src.num_ships == 0:
            return -np.inf

        dist_score = 1 / Planet.distance_between_planets(src, dest)
        foreign_score = 0.25 if dest.owner == game.NEUTRAL else 0.75
        growth_rate_score = (dest.growth_rate - src.growth_rate) / max(src.growth_rate, dest.growth_rate, 1)

        return dist_score + foreign_score + growth_rate_score + ship_dest_score()

    def ships_to_send_in_a_flee(self, source_planet: Planet, dest_planet: Planet) -> int:
        dest_ships_on_arrival = dest_planet.num_ships + \
                                Planet.distance_between_planets(source_planet, dest_planet) * dest_planet.growth_rate
        return min(dest_ships_on_arrival + 1, source_planet.num_ships)

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

                    if best_target[1] > -np.inf:
                        fleet_size = self.ships_to_send_in_a_flee(planet, best_target[0])
                        orders.append(Order(planet.planet_id, best_target[0].planet_id, fleet_size))
                        dests.add(best_target[0])
                except:
                    continue
        except:
            pass

        finally:
            return orders

