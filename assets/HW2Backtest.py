from backtesting import Backtest

class HW2Backtest:
    def __init__(self):
        pass


    def get_best_strategy(buffer, strategy_class):
        # Задаем возможные значения для параметров стратегии
        tema_period_list = [7, 14, 28]
        fastMACD_period_list = [12, 35, 56]
        slowMACD_period_list = [9, 23, 39]
        signalMACD_period_list = [28, 40, 80]

        # Для хранения лучших параметров и лучшего результата
        best_params = None
        best_performance = -float('inf')  # или другое значение, которое имеет смысл для вашей метрики

        # Проходим по всем комбинациям параметров
        for tema_period, fastMACD_period, slowMACD_period, signalMACD_period in itertools.product(tema_period_list, fastMACD_period_list, slowMACD_period_list, signalMACD_period_list):

            # Создаем словарь с текущими параметрами
            params = {
                'tema_period': tema_period,
                'fastMACD_period': fastMACD_period,
                'slowMACD_period': slowMACD_period,
                'signalMACD_period': signalMACD_period
            }

            # Запускаем бэктест с текущими параметрами
            stats = bactest_strategy(buffer.copy(), strategy_class, params)

            # Определяем метрику, по которой будем выбирать лучшую стратегию (например, по профит фактору)
            performance = stats['Profit Factor']  # используйте другую метрику, если нужно

            # Сравниваем с лучшим результатом и сохраняем лучшие параметры
            if performance > best_performance:
                best_performance = performance
                best_params = params

        print(f"Best Performance: {best_performance}")
        print(f"Best Parameters: {best_params}")
        return best_params

if __name__ == "__main__":
    pass