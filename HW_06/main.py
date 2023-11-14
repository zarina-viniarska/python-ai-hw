from collections import Counter
from colorama import Fore

EMPTY_CELL = 0
PLAYER_X = 1
PLAYER_O = -1

board = [
    [9556, 9552, 9552, 9552, 9574, 9552, 9552, 9552, 9574, 9552, 9552, 9552, 9559],
    [9553, 0, 1, 0, 9553, 0, 1, 0, 9553, 0, 1, 0, 9553],
    [9568, 9552, 9552, 9552, 9580, 9552, 9552, 9552, 9580, 9552, 9552, 9552, 9571],
    [9553, 0, 1, 0, 9553, 0, 1, 0, 9553, 0, 1, 0, 9553],
    [9568, 9552, 9552, 9552, 9580, 9552, 9552, 9552, 9580, 9552, 9552, 9552, 9571],
    [9553, 0, 1, 0, 9553, 0, 1, 0, 9553, 0, 1, 0, 9553],
    [9562, 9552, 9552, 9552, 9577, 9552, 9552, 9552, 9577, 9552, 9552, 9552, 9565],
]

# функція виведення гральної дошки
def print_board(values):
    # функція для виведення хрестика або нулика відповідним кольором
    def print_sym(num):
        if num == PLAYER_X:
            print(Fore.MAGENTA + 'X', end='')
        elif num == PLAYER_O:
            print(Fore.CYAN + 'O', end='')
        else:
            print(' ', end='')

    ind = 0
    for i in range (7):
        print('\t', end='')
        for j in range(13):
            if board[i][j] == 0: # потрібно вивести пробіл
                print(' ', end='')
            elif board[i][j] == 1: # на місці 1 має бути клітинка дошки (values)
                print_sym(values[ind])
                ind += 1
            else:
                print(Fore.BLACK + chr(board[i][j]), end='')
        print()
    print(Fore.WHITE, end='')

# функція, що повертає гравця, черга якого ходити
def get_current_player(values):
    counter = Counter(values)
    x_places = counter[PLAYER_X] # к-сть 'X' на дошці 
    o_places = counter[PLAYER_O] # к-сть 'O' на дошці

    if x_places + o_places == 9: # гра закінчена
        return None
    elif x_places > o_places: # черга ходити 'O'
        return PLAYER_O 
    else:
        return PLAYER_X # черга ходити 'X'

# функція, що повертає ходи, які може виконати поточний гравець
def get_actions(values):
    play = get_current_player(values)
    action_list = [(play, i) for i in range(len(values)) if values[i] == EMPTY_CELL]
    return action_list

# функція, що оновлює дошку після ходу
def update_values(values, action):
    (player, move) = action
    values_copy = values.copy()
    values_copy[move] = player
    return values_copy

# перевіряє, чи не настав кінець гри
def is_end_of_game(values):
    # перевіряємо виграшні комбінації
    for i in range(3):
        # по рядках
        if values[3 * i] == values[3 * i + 1] == values[3 * i + 2] != EMPTY_CELL:
            return values[3 * i] # повертаємо переможця
        # по стовпцях
        if values[i] == values[i + 3] == values[i + 6] != EMPTY_CELL:
            return values[i]

    # по діагоналях
    if values[0] == values[4] == values[8] != EMPTY_CELL:
        return values[0]
    if values[2] == values[4] == values[6] != EMPTY_CELL:
        return values[2]

    # якщо немає виграшної комбінації і порожніх клітинок - нічия
    if get_current_player(values) is None:
        return 0
    
    # якщо гра не закінчена
    return None

# функція для оцінки користі для поточної дошки
def utility(values, cost):
    game_state = is_end_of_game(values)
    # якщо гра завершилась
    if game_state != None:
        # повертає результат гри і глибину
        return (game_state, cost)
    # якщо гра не завершилася
    action_list = get_actions(values)
    utils = []
    for action in action_list:
        new_s = update_values(values, action)
        # рекурсивний виклик для усіх можливих ходів іншого гравця
        # зі збільшенням cost (глибини) на один
        utils.append(utility(new_s, cost + 1))

    score = utils[0][0] # початкове значення оцінки
    idx_cost = utils[0][1] # початкове значення користі
    player = get_current_player(values) # отримуємо поточного гравця
    if player == PLAYER_X: # якщо поточний гравець 'X'
        for i in range(len(utils)):
           # якщо оціка ходу в списку utils більша за поточну оцінку, оновлюємо оцінку та вартість
           if utils[i][0] > score:
                score = utils[i][0]
                idx_cost = utils[i][1]
    # якщо поточний гравець 'O'
    else:
        for i in range(len(utils)):
           # якщо оціка ходу в списку utils менша за поточну оцінку, оновлюємо оцінку та вартість
           if utils[i][0] < score:
                score = utils[i][0]
                idx_cost = utils[i][1]
    # повертаємо оцінку та вартість найкращого ходу
    return (score, idx_cost) 

# алгоритм мінімакс для обрахування оптимального ходу
def minimax(values):
    action_list = get_actions(values)
    utils = []
    for action in action_list:
        new_s = update_values(values, action)
        utils.append((action, utility(new_s, 1)))
    # кожний елемент масиву utils містить action, тобто хід
    # разом з його оцінкою та вартістю

    # якщо в масив utils порожній, повертаємо хід за замовчуванням
    if len(utils) == 0:
        return ((0, 0), (0, 0))

    # сортування масиву ходів за спаданням вартості
    sorted_list = sorted(utils, key=lambda l : l[0][1])
    # повертатимемо об'єкт з найменшою оцінкою
    action = min(sorted_list, key = lambda l : l[1])
    return action

if __name__ == '__main__':
    values = [EMPTY_CELL for _ in range(9)]
    print('|---------- ГРА ХРЕСТИКИ-НУЛИКИ ----------|')
    print(" Ти - гравець 'X', комп'ютер - гравець 'O'")
    print_board(values)
    while is_end_of_game(values) is None:
        player = get_current_player(values)
        if player == PLAYER_X:
            try:
                print(f'Твоя черга ходити. Обери клітинку : ', end='')
                move = int(input()) 
            except ValueError:
                print('Потрібно ввести ціле число. Спробуй ще раз.')
                continue
            
            if move < 1 or move > 9:
                print('Число має бути від 1 до 9. Спробуй ще раз.')
                continue
    
            if values[move-1] != EMPTY_CELL: 
                print('Ця клітинка вже зайнята. Спробуй ще раз.')
                continue
    
            values = update_values(values, (1, move-1))
            print_board(values)
        else:
            print("Комп'ютер зробив свій хід.")
            action = minimax(values)
            values = update_values(values, action[0])
            print_board(values)

    winner = is_end_of_game(values) # поверне переможця, або 0 якщо нічия
    if winner == PLAYER_X:
        print('Ти переміг!')
    elif winner == PLAYER_O:
        print("Комп'ютер переміг!")
    else:
        print('Нічия!')