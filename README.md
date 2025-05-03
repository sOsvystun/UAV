# UAV Wind‑Turbine Inspection Suite  

**Методи та засоби динамічного збору візуальних даних про дефекти об’єктів вітроенергетики**  
_Репозиторій супутнього програмного забезпечення до дисертації Сергія Свистуна (Хмельницький національний університет, 2025)._  

---

## 1. Огляд проєкту  

Цей репозиторій містить вихідний код кіберфізичної системи, що описаний у дисертації  
> Свистун С. О. Методи та засоби динамічного збору візуальних даних про дефекти об’єктів вітроенергетики. — Хмельницький, 2025.  

Система забезпечує повністю автономний цикл **«зліт → сканування → аналіз → звіт»** і скорочує час огляду однієї турбіни з 4 год до ≈25 хв за середньої точності виявлення дефектів 92 %. Архітектура кіберфізичної системи складається з трьох взаємопов’язаних компонентів:  

| Підсистема | Мова | Призначення | Ключові посилання у дисертації |
|------------|------|-------------|--------------------------------|
| **VISION_Recognition** | C++20 / Python | Мультиспектральна (RGB + IR) обробка зображень, ансамбль YOLOv8 + Cascade R‑CNN | Розділ 3, § 3.3–3.4 |
| **UAV_server** | Rust 2024 | gRPC‑шина, Kubernetes‑масштабування, потоковий аналіз | Розділ 2, § 2.1 |
| **UAV_Controller** | .NET 8 (MAUI) | UI оператора, телеметрія, планування місій DyTAM | Розділ 2, § 2.2–2.3 |

> 🔗 Завантажити повний текст дисертації: [Свистун_Дисертація](https://nauka.khmnu.edu.ua/category/razovi/)

---

## 2. Функціональні можливості  

* **DyTAM‑маршрути** — адаптивне 3‑D планування обльоту, стійке до вітру 15 м/с.  
* **Blade30‑Thermal Dataset** — 670 пар RGB + IR кадрів (оригінальний [Blade30](https://github.com/cong-yang/Blade30)).  
* **Ансамблевий детектор** — точність F1 = 0.92 на Blade30‑Thermal.  
* **Нечітка оцінка ризику** — індекс критичності з похибкою ≤ 0.2 бала (5‑бальна шкала).  
* **gRPC + MQTT TLS** — безпечне потокове передавання даних у реальному часі.  

---

## 3. Вимоги  

| Компонент | ОС / ПЗ | Версія мінімум |
|-----------|---------|----------------|
| VISION_Recognition | GCC/Clang ‑std=c++20, CMake ≥ 3.30, OpenCV ≥ 4.9, Python ≥ 3.10, PyTorch ≥ 2.2 |  |
| UAV_server | Rust 2024 edition, Cargo, tonic gRPC, OpenCV‑rust binds |  |
| UAV_Controller | Windows 10/11 або macOS 13+, .NET 8 SDK, Visual Studio 2022 17.9+ або VS Code + MAUI extensions |  |

---

## 4. Складання та запуск  

```bash
# 1. Збірка модуля комп'ютерного зору
cd VISION_Recognition && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 2. Збірка серверної шини
cd ../../UAV_server
cargo build --release

# 3. Запуск серверу (gRPC 50051, MQTT 1883)
./target/release/uav_server

# 4. Побудова та запуск UI
open ../UAV_Controller/UAV_Controller.sln  # у Visual Studio
# або
dotnet build ../UAV_Controller -c Release
````

---

## 5. Використання

1. **Запустіть** `uav_server` у кластері Kubernetes або локально.
2. **Підключіть** наземну станцію до дрону (PX4 / ArduPilot).
3. **Відкрийте** UAV\_Controller, виберіть «Нова місія → DyTAM».
4. **Натисніть** «Старт». Після посадки звіт PDF створиться автоматично у `~/UAV_reports/`.

---

## 6. Внесок та відтворюваність

* Експериментальний стенд, конфіг‑файли та скрипти статистичної обробки (`/experiments/`) відтворюють результати розділу 4 дисертації.
* Датасет Blade30‑Thermal доступний за запитом у [issues](https://github.com/sOsvystun/UAV/issues).
* CI workflow `/.github/workflows/test.yml` запускає unit‑тести та перевіряє відтворюваність метрик.

---

## 7. Публікації та цитування

Якщо ви використовуєте програмний код або які-небудь ідеї з дисертації, будь ласка, процитуйте одну з наукових публікацій за темою дисертації:

```
@article{Svystun2025IJC,
  author       = {Svystun, Serhii and Melnychenko, Oleksandr and Radiuk, Pavlo and Savenko, Oleg and Sachenko, Anatoliy and Lysyi, Andrii},
  title        = {Thermal and RGB Images Work Better Together in Wind Turbine Damage Detection},
  journal      = {International Journal of Computing},
  year         = {2025},
  volume       = {23},
  number       = {4},
  pages        = {526--535},
  doi          = {10.47839/ijc.23.4.3752},
  url          = {https://doi.org/10.47839/ijc.23.4.3752},
  note         = {Indexed in Scopus (Q3, SJR)}
}

@article{Svystun2025Energies,
  author       = {Svystun, Serhii and Ścisło, Łukasz and Pawlik, Marcin and Melnychenko, Oleksandr and Radiuk, Pavlo and Savenko, Oleg and Sachenko, Anatoliy},
  title        = {DyTAM: Accelerating Wind Turbine Inspections with Dynamic UAV Trajectory Adaptation},
  journal      = {Energies},
  year         = {2025},
  volume       = {18},
  number       = {7},
  pages        = {1823},
  doi          = {10.3390/en18071823},
  url          = {https://doi.org/10.3390/en18071823},
  note         = {Indexed in Scopus (Q1, SJR)}
}
```

---

## 8. Ліцензія

Код поширюється під ліцензією **MIT**. Дані Blade30‑Thermal — **CC BY‑NC 4.0**.

---

> © 2024‑2025 Serhii Svystun & Contributors
> Хмельницький національний університет, кафедра комп’ютерної інженерії та інформаційних систем