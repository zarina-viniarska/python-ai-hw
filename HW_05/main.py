from colorama import Fore

board = [
    [9556, 9552, 9552, 9552, 9574, 9552, 9552, 9552, 9574, 9552, 9552, 9552, 9559],
    [9553, 0, 1, 0, 9553, 0, 1, 0, 9553, 0, 1, 0, 9553],
    [9568, 9552, 9552, 9552, 9580, 9552, 9552, 9552, 9580, 9552, 9552, 9552, 9571],
    [9553, 0, 1, 0, 9553, 0, 1, 0, 9553, 0, 1, 0, 9553],
    [9568, 9552, 9552, 9552, 9580, 9552, 9552, 9552, 9580, 9552, 9552, 9552, 9571],
    [9553, 0, 1, 0, 9553, 0, 1, 0, 9553, 0, 1, 0, 9553],
    [9562, 9552, 9552, 9552, 9577, 9552, 9552, 9552, 9577, 9552, 9552, 9552, 9565],
]

def print_board(values):
    ind = 0
    for i in range (0, 7):
        print('\t', end='')
        for j in range(0, 13):
            if board[i][j] == 0:
                print(' ', end='')
            elif board[i][j] == 1:
                if values[ind] == 'X':
                    print(Fore.MAGENTA + values[ind], end='')
                else:
                    print(Fore.CYAN + values[ind], end='')
                ind += 1
            else:
                print(Fore.BLACK + chr(board[i][j]), end='')
        print()
    print(Fore.WHITE, end='')

def check_win(player_pos, current_player):
    soln = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7], [2, 5, 8], [3, 6, 9], [1, 5, 9], [3, 5, 7]]
    for x in soln:
        if all(y in player_pos[current_player] for y in x):
            return True       
    return False       
 
def check_draw(player_pos):
    if len(player_pos['X']) + len(player_pos['O']) == 9:
        return True
    return False

def game(current_player):
    values = [' ' for x in range(9)]
    player_pos = {'X':[], 'O':[]}

    while True:
        print_board(values)

        try:
            print(f'Ходить гравець {current_player}. Обери клітинку : ', end='')
            move = int(input()) 
        except ValueError:
            print('Потрібно ввести ціле число. Спробуй ще раз.')
            continue
 
        if move < 1 or move > 9:
            print('Число має бути від 1 до 9. Спробуй ще раз.')
            continue

        if values[move-1] != ' ':
            print('Ця клітинка вже зайнята. Спробуй ще раз.')
            continue
 
        values[move-1] = current_player
        player_pos[current_player].append(move)
 
        if check_win(player_pos, current_player):
            print_board(values)
            print(f'Гравець {current_player} переміг!\n')
            break
        
        if check_draw(player_pos):
            print_board(values)
            print('Нічия!\n')
            break

        if current_player == 'X':
            current_player = 'O'
        else:
            current_player = 'X'

player = 'X'   
while True:
    try:
        print('1 - Грати в хрестики-нулики\n2 - Вихід\nТвій вибір : ', end='')
        choice = int(input())
    except ValueError:
        print('Потрібно ввести ціле число. Спробуй ще раз.')
        continue
    if choice == 1:
        game(player)
        if player == 'X':
            player = 'O'
        else:
            player = 'X'
    elif choice == 2:
        print('До зустрічі в грі!\n')
        break
    else:
        print('Такого пункту меню немає. Спробуй ще раз.')