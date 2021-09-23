import random
from typing import Iterable, List

from courses.planet_wars.planet_wars import Player, PlanetWars, Order, Planet
from courses.planet_wars.tournament import get_map_by_id, run_and_view_battle, TestBot

import pandas as pd

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

class BestBot(Player):

    NAME = 'Arrakis'

    '''
    def get_state(self, game: PlanetWars):
        self.num_ships = game.total_ships_by_owner(game.ME)
        self.opponent_ships = game.total_ships_by_owner(game.ENEMY)
        my_planets = game.get_planets_by_owner(game.ME)
        opponent_planets = game.get_planets_by_owner(game.ENEMY)
        self.total_growth = sum([planet.growth_rate for planet in my_planets])
        self.opponent_growth = sum([planet.growth_rate for planet in opponent_planets])
    '''

    def ships_to_send_in_a_flee(self, source_planet: Planet, dest_planet: Planet) -> int:
        if dest_planet.owner == 0:
            enemy_planet = dest_planet.num_ships
            if(source_planet.num_ships > enemy_planet):
                return enemy_planet+1
            return 0
        if dest_planet.owner == 2:
            on_arrival = dest_planet.num_ships + Planet.distance_between_planets(source_planet,dest_planet)*dest_planet.growth_rate
            if source_planet.num_ships > on_arrival +1:
                return on_arrival
        return 0

    def get_planets_to_attack(self, game: PlanetWars) -> List[Planet]:
        """
        :param game: PlanetWars object representing the map
        :return: The planets we need to attack
        """
        possible_planets = [p for p in game.planets if p.owner != PlanetWars.ME]
        possible_planets = sorted(possible_planets, key = lambda x: x.owner, reverse = True)
        fleets_omw = game.get_fleets_by_owner(game.ME)
        planets_omw = [f.destination_planet_id for f in fleets_omw]
        ans = []
        for planet in possible_planets:
            if planet.planet_id not in planets_omw:
                ans.append(planet)
        return ans

    def attacking_planet_by_radius(self, planet: Planet, radius: int):
        game = self.game
        planets = game.get_planets_by_owner(game.ME)
        for p in planets:
            dist = Planet.distance_between_planets(planet, p)
            if dist == radius:
                return p
        return None

    def stealing_neutral_planets(self, game: PlanetWars):
        fleet_data = game.get_fleets_data_frame()
        fleet_data['destination_owner'] = fleet_data['destination_planet_id'].apply(lambda x: (game.get_planet_by_id(x)).owner)
        fleet_data = fleet_data[fleet_data['owner'] == game.ENEMY]
        fleet_data['destination_planet'] = fleet_data['destination_planet_id'].apply(game.get_planet_by_id)
        fleet_data['destination_planet_ships'] = fleet_data['destination_planet'].apply(lambda x: x.num_ships)
        fleet_data['total_after_conquer'] = fleet_data['num_ships'] - fleet_data['destination_planet_ships']
        fleet_data.set_index('destination_planet_id', inplace=True)
        res = []

        for dest_id, row in fleet_data.iterrows():
            dest_planet = game.get_planet_by_id(dest_id)
            radius = row['turns_remaining']
            attacking_planet = self.attacking_planet_by_radius(dest_planet, radius + 1)
            if (attacking_planet != None):
                res.append(Order(
                attacking_planet,
                dest_id,
                row['total_after_conquer'] + 6))

        return res

    def play_turn(self, game: PlanetWars) -> Iterable[Order]:

        self.game = game
        res = []

        if len(game.get_fleets_data_frame()) > 0:
            res += self.stealing_neutral_planets(game)
            if len(res) > 0:
                return res

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

        how_many = self.ships_to_send_in_a_flee(my_strongest_planet, enemy_or_neutral_weakest_planet)
        if how_many ==0:
            return []

        # (4) Send half the ships from my strongest planet to the weakest planet that I do not own.
        return [Order(
            my_strongest_planet,
            enemy_or_neutral_weakest_planet,
            how_many
        )]

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
    run_and_view_battle(BestBot(), AttackWeakestPlanetFromStrongestSmarterNumOfShipsBot(), map_str)


def test_bot():
    """
    Test AttackWeakestPlanetFromStrongestBot against the 2 other bots.
    Print the battle results data frame and the PlayerScore object of the tested bot.
    So is AttackWeakestPlanetFromStrongestBot worse than the 2 other bots? The answer might surprise you.
    """
    maps = [get_random_map(), get_random_map()]
    player_bot_to_test = BestBot()
    tester = TestBot(
        player=player_bot_to_test,
        competitors=[
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
