import 'dart:io';
import 'dart:convert';
import 'package:llama_cpp_dart/bind/llama_binding.dart';
import 'package:ffi/ffi.dart' as ffi;
import 'dart:ffi';

String libPath = (){
  if(Platform.isAndroid || Platform.isLinux){
    return 'libllama.so';
  } else if(Platform.isIOS || Platform.isMacOS){
    return 'llamalib.dylib';
  } else{
    return 'llamalib.dll';
  }
}();

final DynamicLibrary _dylib = (){
  try {
    return DynamicLibrary.open(libPath);
  }
  catch(e){
    throw Exception('Plataforma não suportada ou biblioteca dinâmica corrompida');
  }
}();
final llamacpp _lib = llamacpp(_dylib);

class LlamaModelParams{
  late llama_model_params _params;

  LlamaModelParams({
    // number of layers to store in VRAM
    int? nGpuLayers,
    // only load the vocabulary, no weights
    bool? vocabOnly,
    // use mmap if possible
    bool? useMmap,
    // force system to keep model in RAM
    bool? useMlock,
    // validate model tensor data
    bool? checkTensors,
    // use extra buffer types (used for weight repacking)
    bool? useExtraBufts
  }){
    _params = _lib.llama_model_default_params();

    if(nGpuLayers != null) _params.n_gpu_layers = nGpuLayers;
    if(vocabOnly != null) _params.vocab_only = vocabOnly;
    if(useMmap != null) _params.use_mmap = useMmap;
    if(useMlock != null) _params.use_mlock = useMlock;
    if(checkTensors != null) _params.check_tensors = checkTensors;
    if(useExtraBufts != null) _params.use_extra_bufts = useExtraBufts;
  }

  llama_model_params getParams(){
    return _params;
  }

  dynamic getParamByKey(String key){
    switch(key){
      case 'devices':
        return _params.devices;
      case 'tensor_buft_overrides':
        return _params.tensor_buft_overrides;
      case 'n_gpu_layers':
        return _params.n_gpu_layers;
      case 'split_mode':
        return _params.split_mode;
      case 'main_gpu':
        return _params.main_gpu;
      case 'tensor_split':
        return _params.tensor_split;
      case 'progress_callback':
        return _params.progress_callback;
      case 'progress_callback_user_data':
        return _params.progress_callback_user_data;
      case 'kv_overrides':
        return _params.kv_overrides;
      case 'vocab_only':
        return _params.vocab_only;
      case 'use_mmap':
        return _params.use_mmap;
      case 'use_mlock':
        return _params.use_mlock;
      case 'check_tensors':
        return _params.check_tensors;
      case 'use_extra_bfts':
        return _params.use_extra_bufts;
      default:
        return null;
    }
  }

  void setParam(String key, dynamic value){
    switch(key){
      case 'devices':
        _params.devices = value;
        break;
      case 'tensor_buft_overrides':
        _params.tensor_buft_overrides = value;
        break;
      case 'n_gpu_layers':
        _params.n_gpu_layers = value;
        break;
      case 'split_mode':
        return;
      case 'main_gpu':
        _params.main_gpu = value;
        break;
      case 'tensor_split':
        _params.tensor_split = value;
        break;
      case 'progress_callback':
        _params.progress_callback = value;
        break;
      case 'progress_callback_user_data':
        _params.progress_callback_user_data = value;
        break;
      case 'kv_overrides':
        _params.kv_overrides = value;
        break;
      case 'vocab_only':
        _params.vocab_only = value;
        break;
      case 'use_mmap':
        _params.use_mmap = value;
        break;
      case 'use_mlock':
        _params.use_mlock = value;
        break;
      case 'check_tensors':
        _params.check_tensors = value;
        break;
      case 'use_extra_bfts':
        _params.use_extra_bufts = value;
        break;
      default:
        return;
    }
  }
}

class LlamaCtxParams{
  late llama_context_params _params;

  LlamaCtxParams({
    // text context, 0 = from model
    int? nCtx,
    // logical maximum batch size that can be submitted to llama_decode
    int? nBatch,
    // physical maximum batch size
    int? nUbatch,
    // max number of sequences
    int? nSeqMax,
    // number of threads to use for generation
    int? nThreads,
    // number of threads to use for batch processing
    int? nThreadsbatch,
    // RoPE base frequency, 0 = from model
    double? ropeFreqBase,
    // RoPE frequency scaling factor, 0 = from model
    double? ropeFreqScale,
    // YaRN extrapolation mix factor, negative = from model
    double? yarnExtFactor,
    // YaRN magnitude scaling factor
    double? yarnAttnFactor,
    // YaRN low correction dim
    double? yarnBetaFast,
    // YaRN high correction dim
    double? yarnBetaSlow,
    // YaRN original context size
    int? yarnOrigCtx,
    // if true, extract embeddings (together with logits)
    bool? embeddings,
    // offload the KQV ops (including the KV cache) to GPU
    bool? offloadKqv,
    // measure performance timings
    bool? noPerf,
    // offload host tensor operations to device
    bool? opOffload,
    // use full-size SWA cache.
    // NOTE: setting to false when n_seq_max > 1 can cause bad performance in some cases
    bool? swaFull,
    // use a unified buffer across the input sequences when computing the attention
    // try to disable when n_seq_max > 1 for improved performance when the sequences do not share a large prefix
    bool? kvUnified
  }){
    _params = _lib.llama_context_default_params();

    if(nCtx != null) _params.n_ctx = nCtx;
    if(nBatch != null) _params.n_batch = nBatch;
    if(nUbatch != null) _params.n_ubatch = nUbatch;
    if(nSeqMax != null) _params.n_seq_max = nSeqMax;
    if(nThreads != null){
      _params.n_threads = nThreads;
    } else{
      _params.n_threads = Platform.numberOfProcessors;
    }
    if(nThreadsbatch != null){
      _params.n_threads_batch = nThreadsbatch;
    } else{
      _params.n_threads = Platform.numberOfProcessors;
    }
    if(ropeFreqBase != null) _params.rope_freq_base = ropeFreqBase;
    if(ropeFreqScale != null) _params.rope_freq_scale = ropeFreqScale;
    if(yarnExtFactor != null) _params.yarn_ext_factor = yarnExtFactor;
    if(yarnAttnFactor != null) _params.yarn_attn_factor = yarnAttnFactor;
    if(yarnBetaFast != null) _params.yarn_beta_fast = yarnBetaFast;
    if(yarnBetaSlow != null) _params.yarn_beta_slow = yarnBetaSlow;
    if(yarnOrigCtx != null) _params.yarn_orig_ctx = yarnOrigCtx;
    if(embeddings != null) _params.embeddings = embeddings;
    if(offloadKqv != null) _params.offload_kqv = offloadKqv;
    if(noPerf != null) _params.no_perf = noPerf;
    if(opOffload != null) _params.op_offload = opOffload;
    if(swaFull != null) _params.swa_full = swaFull;
    if(kvUnified != null) _params.kv_unified = kvUnified;
  }

  llama_context_params getParams(){
    return _params;
  }

  dynamic getParamByKey(String key){
    switch(key){
      case 'n_ctx':
        return _params.n_ctx;
      case 'n_batch':
        return _params.n_batch;
      case 'n_ubatch':
        return _params.n_ubatch;
      case 'n_seq_max':
        return _params.n_seq_max;
      case 'n_threads':
        return _params.n_threads;
      case 'n_threads_batch':
        return _params.n_threads_batch;
      case 'rope_scaling_type':
        return _params.rope_scaling_type;
      case 'pooling_type':
        return _params.pooling_type;
      case 'attention_type':
        return _params.attention_type;
      case 'flash_attn_type':
        return _params.flash_attn_type;
      case 'rope_freq_base':
        return _params.rope_freq_base;
      case 'rope_freq_scale':
        return _params.rope_freq_scale;
      case 'yarn_ext_factor':
        return _params.yarn_ext_factor;
      case 'yarn_attn_factor':
        return _params.yarn_attn_factor;
      case 'yarn_beta_fast':
        return _params.yarn_beta_fast;
      case 'yarn_beta_slow':
        return _params.yarn_beta_slow;
      case 'yarn_orig_ctx':
        return _params.yarn_orig_ctx;
      case 'defrag_thold':
        return _params.defrag_thold;
      case 'cb_eval':
        return _params.cb_eval;
      case 'cb_eval_user_data':
        return _params.cb_eval_user_data;
      case 'type_k':
        return _params.type_k;
      case 'type_v':
        return _params.type_v;
      case 'abort_callback':
        return _params.abort_callback;
      case 'abort_callback_data':
        return _params.abort_callback_data;
      case 'embeddings':
        return _params.embeddings;
      case 'offload_kqv':
        return _params.offload_kqv;
      case 'no_perf':
        return _params.no_perf;
      case 'op_offload':
        return _params.op_offload;
      case 'swa_full':
        return _params.swa_full;
      case 'ku_unified':
        return _params.kv_unified;
      default:
        return null;
    }
  }

  void setParam(String key, dynamic value){
    switch(key){
      case 'n_ctx':
        _params.n_ctx = value;
        break;
      case 'n_batch':
        _params.n_batch = value;
        break;
      case 'n_ubatch':
        _params.n_ubatch = value;
        break;
      case 'n_seq_max':
        _params.n_seq_max = value;
        break;
      case 'n_threads':
        _params.n_threads = value;
        break;
      case 'n_threads_batch':
        _params.n_threads_batch = value;
        break;
      case 'rope_scaling_type':
        return;
      case 'pooling_type':
        return;
      case 'attention_type':
        return;
      case 'flash_attn_type':
        return;
      case 'rope_freq_base':
        _params.rope_freq_base = value;
        break;
      case 'rope_freq_scale':
        _params.rope_freq_scale = value;
        break;
      case 'yarn_ext_factor':
        _params.yarn_ext_factor = value;
        break;
      case 'yarn_attn_factor':
        _params.yarn_attn_factor = value;
        break;
      case 'yarn_beta_fast':
        _params.yarn_beta_fast = value;
        break;
      case 'yarn_beta_slow':
        _params.yarn_beta_slow = value;
        break;
      case 'yarn_orig_ctx':
        _params.yarn_orig_ctx = value;
        break;
      case 'defrag_thold':
        _params.defrag_thold = value;
        break;
      case 'cb_eval':
        _params.cb_eval = value;
        break;
      case 'cb_eval_user_data':
        _params.cb_eval_user_data = value;
        break;
      case 'type_k':
        return;
      case 'type_v':
        return;
      case 'abort_callback':
        _params.abort_callback = value;
        break;
      case 'abort_callback_data':
        _params.abort_callback_data = value;
        break;
      case 'embeddings':
        _params.embeddings = value;
        break;
      case 'offload_kqv':
        _params.offload_kqv = value;
        break;
      case 'no_perf':
        _params.no_perf = value;
        break;
      case 'op_offload':
        _params.op_offload = value;
        break;
      case 'swa_full':
        _params.swa_full = value;
        break;
      case 'ku_unified':
        _params.kv_unified = value;
        break;
      default:
        return;
    }
  }
}

class Llama {
  static final Llama _instance = Llama._internal();
  late LlamaModelParams mParams;
  late LlamaCtxParams cParams;
  late llama_sampler_chain_params sParams;
  late Pointer<llama_vocab> vocab;
  late Pointer<llama_model> model = nullptr;
  late Pointer<llama_context> ctx = nullptr;
  late Pointer<llama_sampler> sampler = nullptr;
  List<String> responce = [];
  bool stopProcess = false;

  factory Llama({
    LlamaModelParams? mParams,
    LlamaCtxParams? cParams
  }){
    if(mParams != null) _instance.mParams = mParams;
    if(cParams != null) _instance.cParams = cParams;

    return _instance;
  }

  Llama._internal(){
    _lib.llama_backend_init();

    mParams = LlamaModelParams();
    cParams = LlamaCtxParams();
    sParams = _lib.llama_sampler_chain_default_params();
  }

  void dispose(){
    if(ctx != nullptr){
      _lib.llama_free(ctx);
    }
    if(model != nullptr){ 
      _lib.llama_model_free(model);
    }
    _lib.llama_backend_free();
  }

  Future<(bool, String)> loadModel(String path) async{
    final pathPtr = path.toNativeUtf8().cast<Char>();
    model = _lib.llama_model_load_from_file(pathPtr, mParams.getParams());
    ffi.malloc.free(pathPtr);
    if(model == nullptr){
      return (false, "Erro ao iniciar o modelo");
    }

    vocab = _lib.llama_model_get_vocab(model);
    if(vocab == nullptr){
      dispose();
      return (false, "Erro ao inicializar o vocab");
    }

    ctx = _lib.llama_init_from_model(model, cParams.getParams());
    if(ctx == nullptr){
      dispose();
      return (false, "Erro ao iniciar o contexto");
    }

    return (true, "Modelo inicializado com sucesso");
  }

  void sendStop(){
    stopProcess = true;
  }

  Stream<String> generateStreamed(String prompt, {
    int nPredict=256,
    double temp=0.8,
    int topK=40,
    double topP=0.95,
    double minP=0.0,
    double penaltyRepeat=1.1,
    double penaltyFreq=0,
    double penaltyPresent=0
  }) async*{
    sampler = _lib.llama_sampler_chain_init(sParams);
    _lib.llama_sampler_chain_add(sampler, _lib.llama_sampler_init_temp(temp));
    _lib.llama_sampler_chain_add(sampler, _lib.llama_sampler_init_top_k(topK));
    _lib.llama_sampler_chain_add(sampler, _lib.llama_sampler_init_top_p(topP, 1));
    _lib.llama_sampler_chain_add(sampler, _lib.llama_sampler_init_min_p(minP, 1));
    _lib.llama_sampler_chain_add(sampler, _lib.llama_sampler_init_penalties(64, penaltyRepeat, penaltyFreq, penaltyPresent));
    _lib.llama_sampler_chain_add(sampler, _lib.llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    if(sampler == nullptr){
      throw Exception('Erro ao iniciar o sampler');
    }

    final tokenList = tokenize(prompt);
    if(tokenList.$1.isEmpty){
      throw Exception('Erro em tokenize() -> ${tokenList.$2}');
    }

    Pointer<llama_token> tokens = ffi.malloc<llama_token>(tokenList.$1.length);
    for(int i=0; i<tokenList.$1.length; i++){
        tokens[i] = tokenList.$1[i];
    }

    var batch = _lib.llama_batch_get_one(tokens, tokenList.$1.length);
    final nextTokenPtr = ffi.malloc<llama_token>();
    final nCtx = _lib.llama_n_ctx(ctx);
    
    try{
      int i = 0;
      while(i < nPredict && !stopProcess){
        final nCtxUsed = _lib.llama_memory_seq_pos_max(_lib.llama_get_memory(ctx), 0) + 1;
        if(nCtxUsed + batch.n_tokens > nCtx){
          throw Exception('Tamanho do contexto excedido');
        }

        if(_lib.llama_decode(ctx, batch) != 0){
          throw Exception('Erro ao decodificar token');
        }

        final nextToken = _lib.llama_sampler_sample(sampler, ctx, -1);
        if(_lib.llama_vocab_is_eog(vocab, nextToken)) break;
        nextTokenPtr.value = nextToken;

        batch = _lib.llama_batch_get_one(nextTokenPtr, 1);

        final result = _detokenize(List.filled(1, nextToken, growable: false));
        if(result.$1 == 0){
          yield result.$2;
        } else{
          break;
        }
      }
    } catch(e){
      throw Exception('Erro ao gerar a resposta: $e');
    }finally{
      stopProcess = false;
      ffi.malloc.free(tokens);
      ffi.malloc.free(nextTokenPtr);
      _lib.llama_sampler_free(sampler);
    }
  }

  Future<String> generate(
    String prompt, {
    bool isIsolated=false,
    int nPredict=256,
    double temp=0.8,
    int topK=40,
    double topP=0.95,
    double minP=0.0,
    double penaltyRepeat=1.1,
    double penaltyFreq=0,
    double penaltyPresent=0
  }) async{
    sampler = _lib.llama_sampler_chain_init(sParams);
    _lib.llama_sampler_chain_add(sampler, _lib.llama_sampler_init_temp(temp));
    _lib.llama_sampler_chain_add(sampler, _lib.llama_sampler_init_top_k(topK));
    _lib.llama_sampler_chain_add(sampler, _lib.llama_sampler_init_top_p(topP, 1));
    _lib.llama_sampler_chain_add(sampler, _lib.llama_sampler_init_min_p(minP, 1));
    _lib.llama_sampler_chain_add(sampler, _lib.llama_sampler_init_penalties(64, penaltyRepeat, penaltyFreq, penaltyPresent));
    _lib.llama_sampler_chain_add(sampler, _lib.llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    if(sampler == nullptr){
      throw Exception('Erro ao iniciar o sampler');
    }

    List<int> acumulated = [];
    
    final tokList = tokenize(prompt);
    if(tokList.$1.isEmpty){
      throw Exception('Erro em tokenize() -> ${tokList.$2}');
    }

    Pointer<llama_token> tokens = ffi.malloc<llama_token>(tokList.$1.length);
    for(int i=0; i<tokList.$1.length; i++){
        tokens[i] = tokList.$1[i];
    }

    llama_batch batch = _lib.llama_batch_get_one(tokens, tokList.$1.length);
    final newToken = ffi.malloc<llama_token>();

    try{
      final nCtx = _lib.llama_n_ctx(ctx);

      int i = 0;
      while(i <= nPredict && !stopProcess){
        final nCtxUsed = _lib.llama_memory_seq_pos_max(_lib.llama_get_memory(ctx), 0) + 1;
        if(nCtxUsed + batch.n_tokens > nCtx){
          throw Exception('Tamanho do contexto excedido');
        }

        final ret = _lib.llama_decode(ctx, batch);
        if(ret != 0){
          throw Exception('Erro ao decodificar tokens');
        }

        final newTokenId = _lib.llama_sampler_sample(sampler, ctx, -1);
        if(_lib.llama_vocab_is_eog(vocab, newTokenId)) break;
        newToken.value = newTokenId;

        batch = _lib.llama_batch_get_one(newToken, 1);

        acumulated.add(newTokenId);
        
        i++;

        if(isIsolated) await Future.delayed(Duration.zero);
      }

      final result = _detokenize(acumulated);
      if(result.$1 != 0){
        throw Exception('Erro ao decodificar tokens para texto -> ${result.$2}');
      }
      return result.$2;
    } catch(e){
      throw Exception('Erro ao gerar a resposta: $e');
    } finally{
      stopProcess = false;
      ffi.malloc.free(tokens);
      ffi.malloc.free(newToken);
      _lib.llama_sampler_free(sampler);
    }
  }

  String formatWithTemplate(dynamic prompt){
    if(prompt is List<Map<String, String>> && prompt.isNotEmpty){
      final size = sizeOf<llama_chat_message>();
      final messages = ffi.malloc<llama_chat_message>(prompt.length * size);
      var offset = 0;
      for(final msg in prompt){
        messages[offset] = ffi.malloc<llama_chat_message>().ref;
        messages[offset].role = msg['role']!.toNativeUtf8().cast<Char>();
        messages[offset].content = msg['content']!.toNativeUtf8().cast<Char>();    

        offset += size;
      }

      final blen = _lib.llama_chat_apply_template(nullptr, messages, prompt.length, false, nullptr, 0);
      if(blen == 0) throw Exception('Não foi possível calcular a alocação do prompt formatado');

      final buffer = ffi.malloc<Uint8>(blen);

      final res = _lib.llama_chat_apply_template(nullptr, messages, prompt.length, false, buffer.cast<Char>(), blen);
      if(res > blen) throw Exception('Não foi possível fazer a alocação do prompt formatado');
      if(res != blen) throw Exception('Erro durante a alocação');
      if(res == 0) throw Exception('Um erro aconteceu na implementação do template');

      final formated = buffer.asTypedList(blen);

      for(var i=prompt.length - 1; i >= 0; i--){
        ffi.malloc.free(messages[offset].content);
        ffi.malloc.free(messages[offset].role);
      }
      ffi.malloc.free(messages);
      ffi.malloc.free(buffer);

      return utf8.decode(formated, allowMalformed: true);
    } else if(prompt is String){
      final llamaMessage = ffi.malloc<llama_chat_message>();
      llamaMessage.ref.role = 'user'.toNativeUtf8().cast<Char>();
      llamaMessage.ref.content = prompt.toNativeUtf8().cast<Char>();

      final blen = 2 * prompt.length;
      final buffer = ffi.malloc<Uint8>(blen);

      final res = _lib.llama_chat_apply_template(nullptr, llamaMessage, 1, false, buffer.cast<Char>(), blen);
      if(res > blen) throw Exception('Não foi possível fazer a alocação do prompt formatado');
      if(res == 0) throw Exception('Um erro aconteceu na implementação do template');

      final formated = buffer.asTypedList(blen);

      ffi.malloc.free(llamaMessage.ref.content);
      ffi.malloc.free(llamaMessage.ref.role);
      ffi.malloc.free(llamaMessage);
      ffi.malloc.free(buffer);

      return utf8.decode(formated, allowMalformed: true);
    } else{
      throw Exception('Formato de entrada inválido');
    }
  }

  (List<int>, String) tokenize(String text){
    final textPtr = text.toNativeUtf8();
    final textLen = textPtr.length;

    final txtTokSize = -_lib.llama_tokenize(vocab, textPtr.cast<Char>(), textLen, nullptr, 0, true, true);
    if(txtTokSize <= 0){
      return ([], 'Erro ao computar quantidade de tokens para alocação');
    }
    Pointer<llama_token> buffer = ffi.malloc<llama_token>(txtTokSize); 

    try {
      final tokCount = _lib.llama_tokenize(
        vocab,
        textPtr.cast<Char>(),
        textLen,
        buffer,
        txtTokSize,
        true,
        true,
      );

      if (tokCount <= 0) {
        return ([], 'Erro ao gerar tokens');
      }
      if (tokCount > txtTokSize) {
        return ([], 'Buffer insuficiente para tokens');
      }

      return (List<int>.generate(tokCount, (i) => buffer[i]), 'Sucesso');
    } catch (e) {
      throw Exception('Error: $e');
    } finally {
      ffi.malloc.free(textPtr);
      ffi.malloc.free(buffer);
    }
  }

  (int, String) _detokenize(List<int> tokens){
    int pieceSize = 0;
    final List<int> n = [];
    for(int pos=0; pos<tokens.length; pos++){
      n.add(-_lib.llama_token_to_piece(vocab, tokens[pos], nullptr, 0, 0, true));
      if(n[pos] <= 0){
        return (1, 'Erro ao computar o tamanho do piece');
      }
      pieceSize += n[pos];
    }
    if(pieceSize <= 0){
      return (1, 'Erro ao computar a alocação do buffer de pieces');
    }
    Pointer<Uint8> buffer = ffi.malloc<Uint8>(pieceSize);
    
    try{
      int offset = 0;
      for(int pos=0; pos<tokens.length; pos++){
        var wrote = _lib.llama_token_to_piece(
          vocab,
          tokens[pos],
          buffer.cast<Char>() + offset,
          n[pos],
          0,
          true
        );

        offset += wrote;
      }
      if(offset != pieceSize){
        return (1, 'O valor de alocação ($pieceSize) é diferente do valor de computação ($offset)');
      }

      final result = buffer.asTypedList(pieceSize);

      return (0, utf8.decode(result, allowMalformed: true));
    } catch(e){
      throw Exception("Erro ao decodificar tokens -> $e");
    } finally{
      ffi.malloc.free(buffer);
    }
  }
}