import os
import sys
import time
import logging
import importlib.util
import importlib.machinery
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

class HotReloadOptimizer:
    def __init__(self, path):
        self.path = path
        self.observers = []
        self.handler = HotReloadHandler(path)
        self.observer = Observer()
        self.observer.schedule(self.handler, path, recursive=True)

    def start(self):
        self.observer.start()
        logger.info('Hot reload observer started')

    def stop(self):
        self.observer.stop()
        self.observer.join()
        logger.info('Hot reload observer stopped')

class HotReloadHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return None
        if not os.path.isfile(event.src_path):
            return None
        if not event.src_path.endswith('.py'):
            return None
        logger.info(f'Reloading {event.src_path}')
        self.reload_module(event.src_path)

    def reload_module(self, path):
        spec = importlib.util.spec_from_file_location('module.name', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules.pop(path, None)
        spec.loader.exec_module(module)

def main():
    path = os.getcwd()
    optimizer = HotReloadOptimizer(path)
    optimizer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        optimizer.stop()
        logger.info('Hot reload optimizer stopped')

if __name__ == '__main__':
    main()
