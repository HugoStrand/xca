struct ScalarStaging {
  std::shared_ptr<DeviceOps> owner;
  nda::vector<scalar_resident_experiment::Weight> host_weights;
  nda::vector<Complex> host_constants;
  nda::vector<std::uintptr_t> host_pointers;
  nda::cuvector<scalar_resident_experiment::Weight> device_weights;
  nda::cuvector<Complex> device_constants;
  nda::cuvector<std::uintptr_t> device_pointers;
#ifdef XCA_EXPERIMENT_SCALAR_STREAMS
  cudaEvent_t weights_uploaded = nullptr, constants_uploaded = nullptr, pointers_uploaded = nullptr;
  ScalarStaging(ScalarStaging const &) = delete;
  ScalarStaging &operator=(ScalarStaging const &) = delete;
  template <typename Host> cudaEvent_t &uploaded_event(Host const &) {
    if constexpr (std::is_same_v<Host, decltype(host_weights)>) return weights_uploaded;
    else if constexpr (std::is_same_v<Host, decltype(host_constants)>) return constants_uploaded;
    else return pointers_uploaded;
  }
  template <typename Host> void prepare(Host const &host) {
    auto event = uploaded_event(host);
    // Only the previous upload must have consumed this host source. Device
    // readers and the next upload are ordered on the same evaluator stream.
    if (event) scalar_edge_experiment::check(cudaEventSynchronize(event));
  }
#endif
  ScalarStaging(std::shared_ptr<DeviceOps> ops, long lanes)
    : owner(std::move(ops)), host_weights(lanes), host_constants(lanes), host_pointers(lanes),
      device_weights(host_weights), device_constants(lanes), device_pointers(lanes) {
    // Weight has nontrivial default initialization. NDA's extent constructor
    // attempts that initialization on the host even for device storage. Copy
    // a host vector instead, and complete this initial default-stream upload
    // before reusing its source on the evaluator stream.
    owner->wait_for_default_uploads();
    owner->finish();
  }
  ~ScalarStaging() {
    // Complete copies/readers before destroying either source or device buffers,
    // including unwinding after a failed launch. Normal operations also finish.
    try { owner->finish(); } catch (...) {}
#ifdef XCA_EXPERIMENT_SCALAR_STREAMS
    for (auto event : {weights_uploaded, constants_uploaded, pointers_uploaded})
      if (event) cudaEventDestroy(event);
#endif
  }
  template <typename Device, typename Host> void upload(Device &device, Host const &host) {
#ifdef XCA_EXPERIMENT_SCALAR_STREAMS
    auto &event = uploaded_event(host);
    if (!event) scalar_edge_experiment::check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
#endif
    scalar_edge_experiment::check(cudaMemcpyAsync(device.data(), host.data(), host.size() * sizeof(*host.data()),
                                                cudaMemcpyHostToDevice, owner->stream()));
#ifdef XCA_EXPERIMENT_SCALAR_STREAMS
    scalar_edge_experiment::check(cudaEventRecord(event, owner->stream()));
#endif
  }
};
