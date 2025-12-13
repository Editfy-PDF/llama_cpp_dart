import 'dart:async';
import 'dart:isolate';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

void main() {
  runApp(const MyApp());
}

/// Estrutura da mensagem que enviamos ao Isolate
class LlamaTask {
  final String modelPath;
  final String prompt;
  final SendPort sendPort;

  LlamaTask({
    required this.modelPath,
    required this.prompt,
    required this.sendPort,
  });
}

/// Função que roda em outro Isolate
Future<void> _llamaWorker(LlamaTask task) async {
  final mParams = LlamaModelParams(nGpuLayers: 99);
  final cParams = LlamaCtxParams(nThreads: 8, nThreadsbatch: 8, offloadKqv: true);
  final llama = Llama(mParams: mParams, cParams: cParams);

  try {
    // Carrega o modelo no isolate
    final isLoaded = await llama.loadModel(task.modelPath);
    if(!isLoaded.$1){
      throw Exception(isLoaded.$2);
    }

    // Gera o texto
    final sw = Stopwatch()..start();
    final result = /*await*/ llama.formatWithTemplate([{'role': 'user', 'content': task.prompt}]);
    sw.stop();
    String debugString = "\t\t\ttime: ${sw.elapsedMilliseconds}ms | tokens: ${llama.tokenize(result).$1.length}";
    // Retorna o resultado
    task.sendPort.send("$result\n$debugString");
  } catch (e) {
    task.sendPort.send('Erro no isolate: $e');
  } finally {
    llama.dispose(); // Libera os recursos
  }
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  String _response = '';
  String? _modelPath;
  bool _isLoading = false;
  bool _modelLoaded = false;

  final TextEditingController _textController = TextEditingController();

  /// Seleciona o arquivo de modelo (.gguf, etc.)
  Future<void> _pickModelFile() async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['gguf', 'bin', 'model'],
      );

      if (result != null && result.files.single.path != null) {
        final path = result.files.single.path!;
        setState(() {
          _modelPath = path;
          _modelLoaded = true;
          _response = 'Modelo selecionado com sucesso!';
        });
      }
    } catch (e) {
      setState(() {
        _response = 'Erro ao selecionar modelo: $e';
      });
    }
  }

  /// Executa a geração em um Isolate separado
  Future<void> _sendText() async {
    if (!_modelLoaded || _modelPath == null) {
      setState(() {
        _response = '⚠️ Carregue um modelo primeiro!';
      });
      return;
    }

    final input = _textController.text.trim();
    if (input.isEmpty) return;

    setState(() {
      _isLoading = true;
      _response = 'Gerando resposta...';
    });

    final receivePort = ReceivePort();

    try {
      await Isolate.spawn(
        _llamaWorker,
        LlamaTask(
          modelPath: _modelPath!,
          prompt: input,
          sendPort: receivePort.sendPort,
        ),
      );

      // Aguarda a resposta do isolate
      final result = await receivePort.first;

      setState(() {
        _response = result.toString();
      });
    } catch (e) {
      setState(() {
        _response = 'Erro ao processar no isolate: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
      receivePort.close();
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('Llama Isolate Example'),
        ),
        body: Padding(
          padding: const EdgeInsets.all(16.0),
          child: SingleChildScrollView(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                // Botão de modelo
                ElevatedButton.icon(
                  onPressed: _isLoading ? null : _pickModelFile,
                  icon: const Icon(Icons.folder_open),
                  label: const Text('Selecionar modelo'),
                ),
                if (_modelPath != null) ...[
                  const SizedBox(height: 8),
                  Text(
                    'Modelo: ${_modelPath!.split('/').last}',
                    style: const TextStyle(fontSize: 14),
                  ),
                ],
                const SizedBox(height: 30),

                TextField(
                  controller: _textController,
                  decoration: const InputDecoration(
                    border: OutlineInputBorder(),
                    labelText: 'Digite seu texto',
                  ),
                ),
                const SizedBox(height: 10),

                ElevatedButton(
                  onPressed: _isLoading ? null : _sendText,
                  child: _isLoading
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Text('Enviar'),
                ),
                const SizedBox(height: 20),

                const Text(
                  'Resposta:',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 10),

                Text(
                  _response.isEmpty ? '(nenhuma resposta ainda)' : _response,
                  textAlign: TextAlign.center,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
