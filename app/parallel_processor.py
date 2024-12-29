from concurrent.futures import ProcessPoolExecutor, as_completed


class ParallelProcessor:
  def __init__(self, process_function):
    self.process_function = process_function

  def execute(self, tasks):
    results = []
    with ProcessPoolExecutor() as executor:
      futures = [executor.submit(self.process_function, *task) for task in tasks]
      
      for futute in as_completed(futures):
        results.extend(futute.result())
    return results
      