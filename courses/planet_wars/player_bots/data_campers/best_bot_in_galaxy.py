import random
from typing import Iterable, List

from courses.planet_wars.planet_wars import Player, PlanetWars, Order, Planet
from courses.planet_wars.tournament import get_map_by_id, run_and_view_battle, TestBot

import pandas as pd


class BestBotInGalaxy(Player):
    NAME = "data_campers"

    def get_ordered_planets(self, game, target_planet_id):
        #The planets this function gets are in IDs
        our_planets = game.get_planets_by_owner(owner=game.ME)
        our_planets = sorted(our_planets, key=lambda x: Planet.distance_between_planets(x, game.get_planet_by_id(target_planet_id)))
        return our_planets

    def get_future_ships(self, planet, time):
        if planet.owner == 0:
            return planet.num_ships
        return planet.growth_rate * time + planet.num_ships

    def calc_ships_needed(self, game, fleet, time_our_fleet_arrivres):
        dest_planet = game.get_planet_by_id(fleet.destination_planet_id)
        incoming_ships = fleet.num_ships
        if fleet.turns_remaining < time_our_fleet_arrivres and dest_planet.owner != game.ENEMY:
            aftermath = incoming_ships - self.get_future_ships(dest_planet, fleet.turns_remaining)
            ribit = dest_planet.growth_rate * (time_our_fleet_arrivres - fleet.turns_remaining + 1)
            needed = aftermath + ribit + 1
        else:
            if dest_planet.owner == game.ME:
                needed = fleet.num_ships - self.get_future_ships(dest_planet, fleet.turns_remaining) + 1
            else:
                needed = 1000000   #high number that will not be sent.
        if needed <= 0:
            needed = 1000000
        return needed


    def place_order(self, game, fleet, available_ships):
        planet_list = self.get_ordered_planets(game, fleet.destination_planet_id)
        for planet in planet_list:
            dist = Planet.distance_between_planets(planet, game.get_planet_by_id(fleet.destination_planet_id)) + 1
            needed = self.calc_ships_needed(game, fleet, dist)
            if needed <= available_ships[planet.planet_id]:
                available_ships[planet.planet_id] -= needed
                return Order(planet, game.get_planet_by_id(fleet.destination_planet_id), needed)

    def check_duplicates(self, game, enemy_fleet):
        our_fleets = game.get_fleets_by_owner(owner=game.ME)
        for our_fleet in our_fleets:
            if enemy_fleet.destination_planet_id == our_fleet.destination_planet_id:
                return True
        return False

    def play_turn(self, game: PlanetWars) -> Iterable[Order]:
        our_planets = game.get_planets_by_owner(owner=game.ME)
        available_ships = {x.planet_id: x.num_ships for x in our_planets}
        orders = []
        for fleet in game.get_fleets_by_owner(owner=game.ENEMY):
            if self.check_duplicates(game, fleet) == False:
                tmp = self.place_order(game, fleet, available_ships)
                if tmp:
                    orders.append(tmp)
        return orders



class AttackWeakestPlanetFromStrongestBot(Player):
    """
    Example of very simple bot - it send flee from its strongest planet to the weakest enemy/neutral planet
    """

    def get_planets_to_attack(self, game: PlanetWars) -> List[Planet]:
        """
        :param game: PlanetWars object representing the map
        :return: The planets we need to attack
        """
        return [p for p in game.planets if p.owner != PlanetWars.ME]

    def ships_to_send_in_a_flee(self, source_planet: Planet, dest_planet: Planet) -> int:
        return source_planet.num_ships // 2

    def play_turn(self, game: PlanetWars) -> Iterable[Order]:
        """
        See player.play_turn documentation.
        :param game: PlanetWars object representing the map - use it to fetch all the planets and flees in the map.
        :return: List of orders to execute, each order sends ship from a planet I own to other planet.
        """
        # (1) If we currently have a fleet in flight, just do nothing.
        if len(game.get_fleets_by_owner(owner=PlanetWars.ME)) >= 1:
            return []

        # (2) Find my strongest planet.
        my_planets = game.get_planets_by_owner(owner=PlanetWars.ME)
        if len(my_planets) == 0:
            return []
        my_strongest_planet = max(my_planets, key=lambda planet: planet.num_ships)

        # (3) Find the weakest enemy or neutral planet.
        planets_to_attack = self.get_planets_to_attack(game)
        if len(planets_to_attack) == 0:
            return []
        enemy_or_neutral_weakest_planet = min(planets_to_attack, key=lambda planet: planet.num_ships)

        # (4) Send half the ships from my strongest planet to the weakest planet that I do not own.
        return [Order(
            my_strongest_planet,
            enemy_or_neutral_weakest_planet,
            self.ships_to_send_in_a_flee(my_strongest_planet, enemy_or_neutral_weakest_planet)
        )]


class AttackEnemyWeakestPlanetFromStrongestBot(AttackWeakestPlanetFromStrongestBot):
    """
    Same like AttackWeakestPlanetFromStrongestBot but attacks only enemy planet - not neutral planet.
    The idea is not to "waste" ships on fighting with neutral planets.

    See which bot is better using the function view_bots_battle
    """

    def get_planets_to_attack(self, game: PlanetWars):
        """
        :param game: PlanetWars object representing the map
        :return: The planets we need to attack - attack only enemy's planets
        """
        return game.get_planets_by_owner(owner=PlanetWars.ENEMY)


class AttackWeakestPlanetFromStrongestSmarterNumOfShipsBot(AttackWeakestPlanetFromStrongestBot):
    """
    Same like AttackWeakestPlanetFromStrongestBot but with smarter flee size.
    If planet is neutral send up to its population + 5
    If it is enemy send most of your ships to fight!

    Will it out preform AttackWeakestPlanetFromStrongestBot? see test_bot function.
    """

    def ships_to_send_in_a_flee(self, source_planet: Planet, dest_planet: Planet) -> int:
        original_num_of_ships = source_planet.num_ships // 2
        if dest_planet.owner == PlanetWars.NEUTRAL:
            if dest_planet.num_ships < original_num_of_ships:
                return dest_planet.num_ships + 5
        if dest_planet.owner == PlanetWars.ENEMY:
            return int(source_planet.num_ships * 0.75)
        return original_num_of_ships


def get_random_map():
    """
    :return: A string of a random map in the maps directory
    """
    random_map_id = random.randrange(1, 100)
    return get_map_by_id(random_map_id)


def view_bots_battle():
    """
    Runs a battle and show the results in the Java viewer

    Note: The viewer can only open one battle at a time - so before viewing new battle close the window of the
    previous one.
    Requirements: Java should be installed on your device.
    """
    map_str = get_random_map()
    run_and_view_battle(BestBotInGalaxy(), AttackWeakestPlanetFromStrongestSmarterNumOfShipsBot(), map_str)


def test_bot():
    """
    Test AttackWeakestPlanetFromStrongestBot against the 2 other bots.
    Print the battle results data frame and the PlayerScore object of the tested bot.
    So is AttackWeakestPlanetFromStrongestBot worse than the 2 other bots? The answer might surprise you.
    """
    maps = [get_random_map(), get_random_map()]
    player_bot_to_test = BestBotInGalaxy()
    tester = TestBot(
        player=player_bot_to_test,
        competitors=[
            #AttackEnemyWeakestPlanetFromStrongestBot()
            AttackWeakestPlanetFromStrongestSmarterNumOfShipsBot()
        ],
        maps=maps
    )
    tester.run_tournament()

    # for a nicer df printing
    pd.set_option('display.max_columns', 30)
    pd.set_option('expand_frame_repr', False)

    print(tester.get_testing_results_data_frame())
    print("\n\n")
    print(tester.get_score_object())

    # To view battle number 4 uncomment the line below
    # tester.view_battle(4)


if __name__ == "__main__":
    test_bot()
    view_bots_battle()
