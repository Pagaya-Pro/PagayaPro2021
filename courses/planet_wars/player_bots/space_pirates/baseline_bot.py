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
    run_and_view_battle(Firstroundstrategy(), AttackEnemyWeakestPlanetFromStrongestBot(), map_str)


def test_bot():
    """
    Test AttackWeakestPlanetFromStrongestBot against the 2 other bots.
    Print the battle results data frame and the PlayerScore object of the tested bot.
    So is AttackWeakestPlanetFromStrongestBot worse than the 2 other bots? The answer might surprise you.
    """
    maps = [get_random_map(), get_random_map()]
    player_bot_to_test = Firstroundstrategy()
    tester = TestBot(
        player=player_bot_to_test,
        competitors=[
            AttackEnemyWeakestPlanetFromStrongestBot(), AttackWeakestPlanetFromStrongestSmarterNumOfShipsBot()
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

class Firstroundstrategy(Player):
    NAME="space_pirates"
    #"accumulate_attack", "combine_attack", "accumulate"
    STATE = "accumulate"
    def get_planets_to_attack(self, game: PlanetWars) -> List[Planet]:
        # if Firstroundstrategy.STATE == 'accumulate_attack':
        if game.turns > 60:
            return [p for p in game.planets if p.owner == PlanetWars.ENEMY]
        return [p for p in game.planets if p.owner != PlanetWars.ME]

    def ships_to_send_in_a_flee(self, source_planet: Planet, dest_planet: Planet) -> int:
        if dest_planet.owner == 0:
            return dest_planet.num_ships + 1
        return dest_planet.num_ships + dest_planet.growth_rate*Planet.distance_between_planets(source_planet,dest_planet) + 1

    # def combine_attack(self, game: PlanetWars) -> Iterable[Order]:
    #     planets_to_attack = game.get_planets_by_owner(owner=PlanetWars.ENEMY)
    #     my_planets = game.get_planets_by_owner(owner=PlanetWars.ME)
    #     planet_to_attack = max(planets_to_attack, key=lambda planet: planet.num_ships)
    #     if len(planets_to_attack) == 0:
    #         return []
    #     planet_scores = {}
    #     for i in my_planets:
    #         ret_dist = Planet.distance_between_planets(planet_to_attack, i)
    #         if (i.growth_rate == 0) or (i.num_ships == 0):
    #             score = ret_dist * ret_dist
    #         else:
    #             score = ret_dist * ret_dist / i.num_ships
    #         planet_scores[i.planet_id] = score
    #     if len(planet_scores) == 0:
    #         return []
    #     # enemy_or_neutral_weakest_score_id = min(planet_scores.keys(), key=lambda planet: planet_scores[planet])
    #
    #     scores_series = pd.Series(planet_scores, name='score')
    #     scores_series.sort_values(inplace=True)
    #     scores_df = scores_series.to_frame().reset_index()
    #     # target_1 = min(planet_scores.keys(), key=lambda planet: planet_scores[planet])
    #
    #     counter = 0
    #     current_sum = 0
    #
    #     for i in range(len(scores_series - 1)):
    #         current_planet = game.get_planet_by_id(scores_df.loc[i, 'index'])
    #
    #         if (current_sum + current_planet.num_ships) < planet_to_attack.num_ships:
    #             counter = counter + 1
    #             current_sum += current_planet.num_ships
    #     if counter == 0:
    #         return []
    #
    #     order_list = []
    #
    #     for i in range(counter):
    #         order_list.append(Order(
    #             scores_df.loc[i, 'index'],  # planet id
    #             planet_to_attack,
    #             self.ships_to_send_in_a_flee(game.get_planet_by_id(scores_df.loc[i, 'index']), planet_to_attack)
    #         ))
    #
    #     return order_list

    def play_turn(self, game: PlanetWars) -> Iterable[Order]:
        # (1) If we currently have a fleet in flight, just do nothing.
        if len(game.get_fleets_by_owner(owner=PlanetWars.ME)) >= 1:
            return []

        # (2) Find my strongest planet.
        my_planets = game.get_planets_by_owner(owner=PlanetWars.ME)
        enemy_planets = game.get_planets_by_owner(owner=PlanetWars.ENEMY)
        my_ships = game.total_ships_by_owner(owner=PlanetWars.ME)
        enemy_ships = game.total_ships_by_owner(owner=PlanetWars.ENEMY)

        if len(my_planets) == 0:
            return []
        my_strongest_planet = max(my_planets, key=lambda planet: planet.num_ships)
        # (3) Find the weakest enemy or neutral planet.

        # if (len(my_planets) < len(enemy_planets)) and (my_ships < enemy_ships):
        #     Firstroundstrategy.STATE = "accumulate"
        # else: #(len(my_planets) < len(enemy_planets)) and (my_ships > enemy_ships):
        #     Firstroundstrategy.STATE = "accumulate_attack"
        # # elif (len(my_planets) > len(enemy_planets)) and (my_ships < enemy_ships):
        # #     Firstroundstrategy.STATE = "combine_attack"
        #     return combine_attack(self, game)
        # else:
        #     Firstroundstrategy.STATE = "combine_attack"
        #     return combine_attack(self, game)

        planets_to_attack = self.get_planets_to_attack(game)
        if len(planets_to_attack) == 0:
            return []
        planet_scores = {}
        for i in planets_to_attack:
            if i.num_ships < my_strongest_planet.num_ships:
                ret_dist = Planet.distance_between_planets(my_strongest_planet,i)
                if i.growth_rate == 0:
                    score = ret_dist * ret_dist * i.num_ships
                else:
                    score = ret_dist*ret_dist*i.num_ships/i.growth_rate
                planet_scores[i.planet_id]=score
        if len(planet_scores)==0:
            return []
        # enemy_or_neutral_weakest_score_id = min(planet_scores.keys(), key=lambda planet: planet_scores[planet])

        scores_series = pd.Series(planet_scores, name='score')
        scores_series.sort_values(inplace=True)
        scores_df = scores_series.to_frame().reset_index()
        # target_1 = min(planet_scores.keys(), key=lambda planet: planet_scores[planet])

        counter = 0
        current_sum = 0

        for i in range(len(scores_series-1)):
            current_planet = game.get_planet_by_id(scores_df.loc[i,'index'])

            if (current_sum + current_planet.num_ships) < my_strongest_planet.num_ships:
                counter = counter + 1
                current_sum += current_planet.num_ships
        if counter == 0:
            return []

        order_list = []

        for i in range(counter):
                my_strongest_planet = max(my_planets, key=lambda planet: planet.num_ships)
                order_list.append(Order(
                my_strongest_planet,
                scores_df.loc[i,'index'], #planet id
                self.ships_to_send_in_a_flee(my_strongest_planet, game.get_planet_by_id(scores_df.loc[i,'index']))
            ))

        return order_list


        #new below


        # return [Order(
        #     my_strongest_planet,
        #     target_1,
        #     self.ships_to_send_in_a_flee(my_strongest_planet, game.get_planet_by_id(enemy_or_neutral_weakest_score_id))
        # ),Order(
        #     my_strongest_planet,
        #     enemy_or_neutral_weakest_score_id,
        #     self.ships_to_send_in_a_flee(my_strongest_planet, game.get_planet_by_id(enemy_or_neutral_weakest_score_id))
        # )]
#end

if __name__ == "__main__":
    test_bot()
    #view_bots_battle()
