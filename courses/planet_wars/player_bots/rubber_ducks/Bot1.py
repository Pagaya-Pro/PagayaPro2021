import random
from typing import Iterable, List

from courses.planet_wars.planet_wars import Player, PlanetWars, Order, Planet
from courses.planet_wars.tournament import get_map_by_id, run_and_view_battle, TestBot

import pandas as pd


class Bot1(Player):
    def help_planets_need_help(self, game: PlanetWars) -> Iterable[Order]:
        orders = []

        planets_need_help = []

        for planet in game.get_planets_by_owner(PlanetWars.ME):
            # already sent help
            for fleet in game.get_fleets_by_owner(PlanetWars.ME):
                if fleet.destination_planet_id == planet.planet_id:
                    continue

            attack_size = 0
            turns_to_first_attack = 10000
            for fleet in game.get_fleets_by_owner(PlanetWars.ENEMY):
                if fleet.destination_planet_id == planet.planet_id:
                    attack_size += fleet.num_ships
                    turns_to_first_attack = min(turns_to_first_attack, fleet.turns_remaining)

            if attack_size == 0:
                continue

            planet_size_at_first_attack = turns_to_first_attack * planet.growth_rate + planet.num_ships
            planet_need = attack_size - planet_size_at_first_attack

            if planet_need <= 0:
                continue

            planets_need_help.append((planet, planet_need, turns_to_first_attack))

        for need_help, planet_need, turns_to_first_attack in planets_need_help:
            helpers = []
            size_of_help = 0

            for help_planet in game.get_planets_by_owner(PlanetWars.ME):
                if help_planet in planets_need_help:
                    continue
                if Planet.distance_between_planets(help_planet, need_help) >= turns_to_first_attack:
                    continue
                helpers.append((help_planet, min(0.65 * help_planet.num_ships, planet_need)))
                size_of_help += min(0.65 * help_planet.num_ships, planet_need)

            if size_of_help >= planet_need:
                for help_planet, amount in helpers:
                    orders.append(Order(help_planet, need_help, amount))

        return orders

    def go_to_closest(self, game: PlanetWars) ->Iterable[Order]:
        minimum_in_planet = 20
        planets_i_sent_to = [f.destination_planet_id for f in game.get_fleets_by_owner(owner=PlanetWars.ME)]
        planets = game.get_planets_by_owner(owner=PlanetWars.NEUTRAL) + game.get_planets_by_owner(
            owner=PlanetWars.ENEMY)
        planets = [p for p in planets if p.planet_id not in planets_i_sent_to]
        my_planets = [p for p in game.get_planets_by_owner(owner=PlanetWars.ME) if p.num_ships > minimum_in_planet]
        orders = []
        for mp in my_planets:
            to_delete = set()
            free_ships = mp.num_ships - minimum_in_planet
            planets.sort(key=lambda p: p.num_ships * (5 - Planet.distance_between_planets(mp, p)), reverse=True)
            for p in planets:
                neutral = p.owner == PlanetWars.NEUTRAL
                threshold = p.num_ships if neutral else p.num_ships + (Planet.distance_between_planets(mp, p) * p.growth_rate)
                if threshold < free_ships:
                    free_ships -= p.num_ships + 1
                    orders.append(Order(
                        mp,
                        p,
                        p.num_ships + 1)
                    )
                    planets_i_sent_to.append(p.planet_id)
                    to_delete.add(p)
            planets = list(set(planets) - to_delete)
        return orders

    def play_turn(self, game: PlanetWars) -> Iterable[Order]:
        orders = []
        orders += self.help_planets_need_help(game)
        if not orders:
            return self.go_to_closest(game)
        return orders
