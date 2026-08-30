docker logs mbx-vllm -f 2>&1 \
  | grep --line-buffered -vE '"(GET|HEAD) /(v1/models|metrics|health|server_info|get_server_info|model_info|get_model_info)|is deprecated and will be removed' \
  | grcat ~/.grc/vllm.conf
