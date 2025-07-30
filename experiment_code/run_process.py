import subprocess
import shlex

command = "qsub script.sh -g tga-cddlab"
num_executions = 18

print(f"'{command}' を {num_executions} 回実行します。")

for i in range(num_executions):
    try:
        print(f"実行 {i + 1}/{num_executions} 回目")
        args = shlex.split(command)
        result = subprocess.run(args, check=True, capture_output=True, text=True)
        print("標準出力:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"エラーが発生しました: {e}")
        print("標準エラー出力:")
        print(e.stderr)
    except FileNotFoundError:
        print("エラー: 'qsub'コマンドが見つかりません。パスが通っているか確認してください。")
        break

print("すべてのコマンドの実行が完了しました。")
