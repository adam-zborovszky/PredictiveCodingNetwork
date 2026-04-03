#include "ui/app_window.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

#include <cstdio>
#include <chrono>
#include <algorithm>
#include <string>
#include <vector>
#include <functional>

static void glfwErrorCallback(int error, const char* description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

bool runAppWindow(const std::string& gpu_name, TrainingState& state,
                  StartTrainingFunc start_training) {
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit()) {
        fprintf(stderr, "ERROR: Failed to initialize GLFW\n");
        return false;
    }

    // OpenGL 3.3 core
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

    std::string title = "PCN MNIST Trainer — " + gpu_name;
    GLFWwindow* window = glfwCreateWindow(1600, 900, title.c_str(), nullptr, nullptr);
    if (!window) {
        fprintf(stderr, "ERROR: Failed to create GLFW window\n");
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // vsync

    // ImGui setup
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.FontGlobalScale = 1.0f;

    // Dark theme
    ImGui::StyleColorsDark();

    // Font size 14px
    io.Fonts->AddFontDefault();
    ImFontConfig font_cfg;
    font_cfg.SizePixels = 14.0f;
    io.Fonts->AddFontDefault(&font_cfg);

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    auto start_time = std::chrono::steady_clock::now();
    float best_accuracy = 0.0f;
    int best_network_idx = -1;

    // Energy history snapshot for plotting (to avoid locking every frame)
    std::vector<float> energy_snapshot;
    int snapshot_frame = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Full-screen window
        ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->Pos);
        ImGui::SetNextWindowSize(viewport->Size);
        ImGui::Begin("PCN MNIST Trainer", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoBringToFrontOnFocus);

        float total_width = ImGui::GetContentRegionAvail().x;
        float total_height = ImGui::GetContentRegionAvail().y;

        // ─── Left Panel (30%) — Config & Control ────────────────────────
        ImGui::BeginChild("LeftPanel", ImVec2(total_width * 0.30f, total_height), true);
        ImGui::Text("Network Configurations");
        ImGui::Separator();

        for (size_t i = 0; i < state.configs.size(); i++) {
            auto& cfg = state.configs[i];
            ImGui::PushID(static_cast<int>(i));

            // Status color
            NetworkStatus status = NetworkStatus::Pending;
            {
                std::lock_guard<std::mutex> lock(state.results_mutex);
                if (i < state.results.size())
                    status = state.results[i].status;
            }

            ImVec4 status_color;
            const char* status_text;
            switch (status) {
                case NetworkStatus::Pending:
                    status_color = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
                    status_text = "Pending";
                    break;
                case NetworkStatus::Training:
                    status_color = ImVec4(1.0f, 0.8f, 0.0f, 1.0f);
                    status_text = "Training";
                    break;
                case NetworkStatus::Done:
                    status_color = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
                    status_text = "Done";
                    break;
                case NetworkStatus::Error:
                    status_color = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
                    status_text = "Error";
                    break;
            }

            ImGui::TextColored(status_color, "[%s]", status_text);
            ImGui::SameLine();
            ImGui::Text("%s", cfg.name.c_str());

            // Layer sizes
            ImGui::Text("  Layers: ");
            ImGui::SameLine();
            for (size_t l = 0; l < cfg.layers.size(); l++) {
                if (l > 0) { ImGui::SameLine(); ImGui::Text("->"); ImGui::SameLine(); }
                ImGui::Text("%d", cfg.layers[l]);
                if (l < cfg.layers.size() - 1) ImGui::SameLine();
            }

            ImGui::Text("  LR: %.4f  Steps: %d  Epochs: %d",
                        cfg.learning_rate, cfg.inference_steps, cfg.epochs);
            ImGui::Spacing();

            ImGui::PopID();
        }

        ImGui::Separator();

        bool is_running = state.running.load();
        if (!is_running) {
            if (ImGui::Button("Train All", ImVec2(-1, 40))) {
                start_training(state);
            }
        } else {
            if (ImGui::Button("Stop", ImVec2(-1, 40))) {
                state.stop_requested.store(true);
            }
        }

        ImGui::EndChild();

        ImGui::SameLine();

        // ─── Center Panel (50%) — Live Training ─────────────────────────
        ImGui::BeginChild("CenterPanel", ImVec2(total_width * 0.50f, total_height), true);
        ImGui::Text("Live Training");
        ImGui::Separator();

        if (state.running.load()) {
            int cur_net = state.current_network.load();
            int cur_epoch = state.current_epoch.load();
            int tot_epochs = state.total_epochs.load();
            int cur_sample = state.current_sample.load();
            int tot_samples = state.total_samples.load();
            float cur_energy = state.current_energy.load();

            ImGui::Text("Network: %s", state.current_network_name.c_str());
            ImGui::Text("Progress: Epoch %d/%d, Sample %d/%d",
                        cur_epoch + 1, tot_epochs, cur_sample + 1, tot_samples);

            // Progress bar
            float progress = 0.0f;
            if (tot_epochs > 0 && tot_samples > 0) {
                progress = (cur_epoch * tot_samples + cur_sample) /
                           static_cast<float>(tot_epochs * tot_samples);
            }
            ImGui::ProgressBar(progress, ImVec2(-1, 20));

            ImGui::Text("Current Energy: %.4f", cur_energy);

            // Elapsed and ETA
            auto now = std::chrono::steady_clock::now();
            float elapsed_sec = std::chrono::duration<float>(now - start_time).count();
            ImGui::Text("Elapsed: %.1fs", elapsed_sec);
            if (progress > 0.01f) {
                float eta = elapsed_sec * (1.0f - progress) / progress;
                ImGui::SameLine();
                ImGui::Text("  ETA: %.1fs", eta);
            }

            // Update energy snapshot every few frames
            snapshot_frame++;
            if (snapshot_frame % 5 == 0) {
                std::lock_guard<std::mutex> lock(state.history_mutex);
                energy_snapshot = state.energy_history;
            }

            // Plot last 500 energy values
            if (!energy_snapshot.empty()) {
                int plot_count = static_cast<int>(energy_snapshot.size());
                int offset = 0;
                if (plot_count > 500) {
                    offset = plot_count - 500;
                    plot_count = 500;
                }

                // Compute average
                float avg = 0.0f;
                for (int i = offset; i < offset + plot_count; i++) {
                    avg += energy_snapshot[i];
                }
                avg /= plot_count;

                char overlay[64];
                snprintf(overlay, sizeof(overlay), "Avg: %.4f", avg);

                ImGui::PlotLines("Energy", energy_snapshot.data() + offset,
                                 plot_count, 0, overlay, FLT_MAX, FLT_MAX,
                                 ImVec2(-1, 200));

                ImGui::Text("Epoch Average Energy: %.4f", avg);
            }
        } else {
            ImGui::TextDisabled("No training in progress. Click 'Train All' to start.");
        }

        ImGui::EndChild();

        ImGui::SameLine();

        // ─── Right Panel (20%) — Results ────────────────────────────────
        ImGui::BeginChild("RightPanel", ImVec2(0, total_height), true);
        ImGui::Text("Results");
        ImGui::Separator();

        {
            std::lock_guard<std::mutex> lock(state.results_mutex);

            best_accuracy = 0.0f;
            best_network_idx = -1;

            for (size_t i = 0; i < state.results.size(); i++) {
                auto& r = state.results[i];
                if (r.status != NetworkStatus::Done) continue;

                if (r.accuracy > best_accuracy) {
                    best_accuracy = r.accuracy;
                    best_network_idx = static_cast<int>(i);
                }
            }

            for (size_t i = 0; i < state.results.size(); i++) {
                auto& r = state.results[i];
                if (r.status != NetworkStatus::Done) continue;

                bool is_best = (static_cast<int>(i) == best_network_idx);

                if (is_best) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 1.0f, 0.5f, 1.0f));
                }

                ImGui::Text("%s%s: %.2f%%",
                           is_best ? "* " : "  ",
                           r.name.c_str(), r.accuracy * 100.0f);

                if (is_best) {
                    ImGui::PopStyleColor();
                }

                // Per-class accuracy table
                if (ImGui::TreeNode(("Details##" + r.name).c_str())) {
                    ImGui::Columns(4, nullptr, true);
                    ImGui::SetColumnWidth(0, 50);
                    ImGui::SetColumnWidth(1, 70);
                    ImGui::SetColumnWidth(2, 70);

                    ImGui::Text("Digit"); ImGui::NextColumn();
                    ImGui::Text("Correct"); ImGui::NextColumn();
                    ImGui::Text("Total"); ImGui::NextColumn();
                    ImGui::Text("Acc%%"); ImGui::NextColumn();
                    ImGui::Separator();

                    for (int d = 0; d < 10; d++) {
                        ImGui::Text("%d", d); ImGui::NextColumn();
                        ImGui::Text("%d", r.per_class_correct[d]); ImGui::NextColumn();
                        ImGui::Text("%d", r.per_class_total[d]); ImGui::NextColumn();
                        float pct = (r.per_class_total[d] > 0) ?
                            100.0f * r.per_class_correct[d] / r.per_class_total[d] : 0.0f;
                        ImGui::Text("%.1f", pct); ImGui::NextColumn();
                    }

                    ImGui::Columns(1);
                    ImGui::TreePop();
                }

                ImGui::Spacing();
            }
        }

        ImGui::EndChild();
        ImGui::End();

        // Render
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return true;
}
