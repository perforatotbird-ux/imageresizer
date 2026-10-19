"""Stdlib menu: user input is never interpolated into a cmd.exe command."""
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def number(prompt, default, low=1, high=100000):
    while True:
        value = input(f'{prompt} [{default}]: ').strip() or str(default)
        if value.isdecimal() and low <= int(value) <= high:
            return str(int(value))
        print(f'Enter an integer in {low}..{high}.')

def pick(prompt, values, default):
    print(prompt + ': ' + ', '.join(values))
    while True:
        value = input(f'[{default}]: ').strip() or default
        if value in values:
            return value
        print('Choose one of the displayed values.')

def main():
    while True:
        print('\nbatchresize: input/ -> output/\n1 Light  2 Medium  3 Strong  4 Extreme'
              '\n5 Thumbnails  6 Custom  7 Benchmark  8 Environment  9 GUI  0 Exit')
        choice = input('Select: ').strip()
        if choice == '0':
            return 0
        if choice == '8':
            args = ['--doctor']
        elif choice == '9':
            args = ['--gui']
        elif choice == '7':
            args = ['--benchmark', '--benchmark-repeats', number('Repeats', 3, 1, 100)]
        elif choice in ('1', '2', '3', '4', '5'):
            args = ['--quality', {'1':'light','2':'medium','3':'strong','4':'extreme','5':'strong'}[choice]]
            if choice == '5':
                args += ['--size', 'thumb_256', '--mode', 'cover']
            args += ['--workers', number('Threads', 8, 1, 64)]
        elif choice == '6':
            quality = pick('Compression', ['light','medium','strong','extreme'], 'medium')
            size = pick('Size', ['thumb_256','small_512','web_1024','large_1600','custom'], 'web_1024')
            args = ['--quality', quality, '--size', size]
            if size == 'custom':
                args += ['--width', number('Width', 1600), '--height', number('Height', 1600)]
            args += ['--mode', pick('Mode', ['contain','cover'], 'contain'),
                     '--workers', number('Threads', 8, 1, 64),
                     '--backend', pick('Backend', ['auto','gpu','hybrid','cpu'], 'auto'),
                     '--batch-size', number('Batch size', 8, 1, 128)]
        else:
            print('Choose 0..9.'); continue
        if choice not in ('8', '9') and input('Run now? [y/N]: ').strip().lower() != 'y':
            continue
        code = subprocess.call([sys.executable, '-X', 'utf8', '-m', 'batchresize', *args], cwd=ROOT)
        print(f'Exit code: {code}')

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (EOFError, KeyboardInterrupt):
        raise SystemExit(130)
