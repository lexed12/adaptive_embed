import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import math

class AdaptiveSteganography:
    def __init__(self, image_path, message):
        self.original_image = Image.open(image_path).convert('RGB')
        self.image_array = np.array(self.original_image)
        self.message = message
        self.height, self.width = self.image_array.shape[:2]
        
        # Параметры блоков
        self.block_size = 8
        self.blocks_y = self.height // self.block_size
        self.blocks_x = self.width // self.block_size
        
        # Предвычисление энтропии для всех блоков
        self.block_entropies = self._calculate_block_entropies()
        
        # Сортировка блоков по убыванию энтропии
        self.sorted_blocks = self._sort_blocks_by_entropy()
        
        # Максимальная теоретическая емкость
        self.max_capacity = self._calculate_max_capacity()
        
        # Генерация всех 25 уровней
        self.stego_images = self._generate_stego_levels()
    
    def _calculate_entropy(self, block):
        hist, _ = np.histogram(block.flatten(), bins=256, range=(0, 256))
        hist = hist[hist > 0]
        probabilities = hist / np.sum(hist)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def _calculate_block_entropies(self):
        entropies = np.zeros((self.blocks_y, self.blocks_x))
        
        for y in range(self.blocks_y):
            for x in range(self.blocks_x):
                y_start = y * self.block_size
                y_end = (y + 1) * self.block_size
                x_start = x * self.block_size
                x_end = (x + 1) * self.block_size
                
                block = self.image_array[y_start:y_end, x_start:x_end]
                
                block_entropy = 0
                for channel in range(3):
                    channel_block = block[:, :, channel]
                    block_entropy += self._calculate_entropy(channel_block)
                
                entropies[y, x] = block_entropy / 3
        
        return entropies
    
    def _sort_blocks_by_entropy(self):
        blocks = []
        for y in range(self.blocks_y):
            for x in range(self.blocks_x):
                blocks.append({
                    'y': y,
                    'x': x,
                    'entropy': self.block_entropies[y, x]
                })
        
        blocks.sort(key=lambda b: b['entropy'], reverse=True)
        return blocks
    
    def _calculate_max_capacity(self):
        total_blocks = self.blocks_y * self.blocks_x
        bits_per_block = self.block_size * self.block_size * 3 * 8
        max_bits = total_blocks * bits_per_block
        return max_bits // 8
    
    def _get_channels_for_level(self, level_percent):
        if level_percent < 0.33:
            return [2]  # Только синий
        elif level_percent < 0.66:
            return [2, 0]  # Синий и красный
        else:
            return [0, 1, 2]  # Все каналы
    
    def _get_bits_per_channel(self, level_percent):
        if level_percent < 0.125:
            return 1
        elif level_percent < 0.25:
            return 2
        elif level_percent < 0.375:
            return 3
        elif level_percent < 0.5:
            return 4
        elif level_percent < 0.625:
            return 5
        elif level_percent < 0.75:
            return 6
        elif level_percent < 0.875:
            return 7
        else:
            return 8
    
    def _embed_message(self, level):
        stego_array = self.image_array.copy()
        
        # Количество блоков для встраивания пропорционально уровню
        num_blocks = int(len(self.sorted_blocks) * level)
        
        # Определяем параметры встраивания
        channels_to_use = self._get_channels_for_level(level)
        bits_per_channel = self._get_bits_per_channel(level)
        
        # Преобразуем сообщение в биты
        message_bytes = self.message.encode('utf-8')
        message_bits = ''.join(format(byte, '08b') for byte in message_bytes)
        
        # Рассчитываем необходимое количество бит
        bits_per_pixel = len(channels_to_use) * bits_per_channel
        total_bits_needed = num_blocks * self.block_size * self.block_size * bits_per_pixel
        
        # Дублируем сообщение при необходимости
        if len(message_bits) < total_bits_needed:
            repeats = total_bits_needed // len(message_bits) + 1
            message_bits = (message_bits * repeats)[:total_bits_needed]
        
        bit_index = 0
        
        # Встраивание в блоки с самой высокой энтропии
        for block_info in self.sorted_blocks[:num_blocks]:
            y = block_info['y']
            x = block_info['x']
            
            y_start = y * self.block_size
            y_end = (y + 1) * self.block_size
            x_start = x * self.block_size
            x_end = (x + 1) * self.block_size
            
            for i in range(self.block_size):
                for j in range(self.block_size):
                    for channel in channels_to_use:
                        if bit_index < len(message_bits):
                            pixel_value = stego_array[y_start + i, x_start + j, channel]
                            
                            bits_to_embed = message_bits[bit_index:bit_index + bits_per_channel]
                            if len(bits_to_embed) == bits_per_channel:
                                value_to_embed = int(bits_to_embed, 2)
                                
                                if bits_per_channel == 8:
                                    stego_array[y_start + i, x_start + j, channel] = value_to_embed
                                else:
                                    mask = (0xFF << bits_per_channel) & 0xFF
                                    stego_array[y_start + i, x_start + j, channel] = (pixel_value & mask) | value_to_embed
                                
                                bit_index += bits_per_channel
        
        return Image.fromarray(stego_array.astype('uint8'))
    
    def _generate_stego_levels(self):
        levels = 24
        stego_images = []
        
        for i in range(levels + 1):
            level = i / levels
            print(f"Генерация уровня {i}/{levels}...")
            stego_image = self._embed_message(level)
            stego_images.append(stego_image)
        
        return stego_images
    
    def get_image_for_level(self, level_index):
        return self.stego_images[level_index]
    
    def calculate_psnr(self, original, stego):
        if original.size != stego.size:
            stego = stego.resize(original.size, Image.Resampling.LANCZOS)
        
        original_array = np.array(original)
        stego_array = np.array(stego)
        
        mse = np.mean((original_array - stego_array) ** 2)
        if mse == 0:
            return float('inf')
        
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        return psnr


class ImageViewer(ttk.Frame):
    """Виджет для отображения изображения с прокруткой в реальном масштабе"""
    def __init__(self, parent, title):
        super().__init__(parent)
        
        # Заголовок
        title_label = ttk.Label(self, text=title, font=('Arial', 12, 'bold'))
        title_label.pack(pady=5)
        
        # Создаем фрейм с прокруткой
        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas для отображения изображения
        self.canvas = tk.Canvas(canvas_frame, bg='#2b2b2b', highlightthickness=1, 
                                highlightbackground='#555')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Полосы прокрутки
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        h_scrollbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Настройка canvas
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Привязка событий для масштабирования колесиком мыши
        self.canvas.bind('<MouseWheel>', self._on_mousewheel)
        self.canvas.bind('<Control-MouseWheel>', self._on_ctrl_mousewheel)
        
        # Переменные для масштабирования
        self.scale_factor = 1.0
        self.image = None
        self.image_tk = None
        self.image_id = None
        self.original_image = None
        
        # Информация о размере
        self.info_label = ttk.Label(self, text="Размер: -", font=('Arial', 9))
        self.info_label.pack(pady=2)
    
    def set_image(self, pil_image):
        """Установка изображения для отображения"""
        self.original_image = pil_image
        self.original_size = pil_image.size
        self.scale_factor = 1.0
        self._update_display()
        self.info_label.config(text=f"Размер: {self.original_size[0]}x{self.original_size[1]} пикселей")
    
    def _update_display(self):
        """Обновление отображения с текущим масштабом"""
        if self.original_image is None:
            return
        
        # Масштабируем изображение
        new_size = (int(self.original_size[0] * self.scale_factor),
                   int(self.original_size[1] * self.scale_factor))
        
        if new_size[0] > 0 and new_size[1] > 0:
            scaled_image = self.original_image.resize(new_size, Image.Resampling.NEAREST)
            self.image_tk = ImageTk.PhotoImage(scaled_image)
            
            # Обновляем canvas
            self.canvas.delete("all")
            self.image_id = self.canvas.create_image(0, 0, image=self.image_tk, anchor=tk.NW)
            
            # Настраиваем область прокрутки
            self.canvas.configure(scrollregion=(0, 0, new_size[0], new_size[1]))
    
    def _on_mousewheel(self, event):
        """Обработка колесика мыши для вертикальной прокрутки"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def _on_ctrl_mousewheel(self, event):
        """Обработка Ctrl+колесико для масштабирования"""
        if event.delta > 0:
            self.scale_factor *= 1.1
        else:
            self.scale_factor *= 0.9
        
        # Ограничиваем масштаб
        self.scale_factor = max(0.1, min(5.0, self.scale_factor))
        
        self._update_display()
        
        # Обновляем информацию о масштабе
        self.info_label.config(
            text=f"Размер: {self.original_size[0]}x{self.original_size[1]} пикселей | Масштаб: {self.scale_factor:.1f}x"
        )


class SteganographyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Адаптивная стеганография на основе энтропии")
        self.root.geometry("1200x800")
        
        self.stego_system = None
        self.current_level = 0
        
        self.setup_ui()
    
    def setup_ui(self):
        # Создаем вкладки
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Вкладка 1: Оригинальное изображение
        self.original_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.original_tab, text="Оригинал")
        self.setup_original_tab()
        
        # Вкладка 2: Стего-изображение с ползунком
        self.stego_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.stego_tab, text="Стего-изображение")
        self.setup_stego_tab()
        
        # Вкладка 3: Информация и статистика
        self.info_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.info_tab, text="Информация")
        self.setup_info_tab()
        
        # Нижняя панель с общими элементами управления
        self.setup_control_panel()
    
    def setup_original_tab(self):
        """Вкладка с оригинальным изображением"""
        # Кнопка загрузки
        btn_frame = ttk.Frame(self.original_tab)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.load_btn = ttk.Button(btn_frame, text="Загрузить изображение", 
                                   command=self.load_image)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        self.file_label = ttk.Label(btn_frame, text="Файл не выбран")
        self.file_label.pack(side=tk.LEFT, padx=10)
        
        # Просмотрщик оригинального изображения
        self.original_viewer = ImageViewer(self.original_tab, "Оригинальное изображение")
        self.original_viewer.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def setup_stego_tab(self):
        """Вкладка со стего-изображением и ползунком"""
        # Верхняя панель с ползунком
        slider_frame = ttk.LabelFrame(self.stego_tab, text="Уровень встраивания", padding=10)
        slider_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Ползунок
        slider_controls = ttk.Frame(slider_frame)
        slider_controls.pack(fill=tk.X, pady=5)
        
        ttk.Label(slider_controls, text="Меньше").pack(side=tk.LEFT, padx=5)
        
        self.level_var = tk.DoubleVar(value=0)
        self.level_slider = ttk.Scale(
            slider_controls,
            from_=0, to=24,
            orient=tk.HORIZONTAL,
            variable=self.level_var,
            command=self.on_slider_changed,
            state='disabled'
        )
        self.level_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        ttk.Label(slider_controls, text="Больше").pack(side=tk.LEFT, padx=5)
        
        # Метка с текущим уровнем
        self.level_label = ttk.Label(slider_frame, text="Уровень: 0/24", font=('Arial', 10, 'bold'))
        self.level_label.pack(pady=5)
        
        # Индикатор этапов
        stages_frame = ttk.Frame(slider_frame)
        stages_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(stages_frame, text="[Синий]", foreground='blue').pack(side=tk.LEFT, expand=True)
        ttk.Label(stages_frame, text="[Синий+Красный]", foreground='purple').pack(side=tk.LEFT, expand=True)
        ttk.Label(stages_frame, text="[Все каналы]", foreground='black').pack(side=tk.LEFT, expand=True)
        
        # Детальная информация
        self.detail_label = ttk.Label(slider_frame, text="", font=('Arial', 9))
        self.detail_label.pack(pady=2)
        
        # Просмотрщик стего-изображения
        self.stego_viewer = ImageViewer(self.stego_tab, "Стего-изображение")
        self.stego_viewer.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def setup_info_tab(self):
        """Вкладка с информацией и статистикой"""
        # Основная информация
        info_frame = ttk.LabelFrame(self.info_tab, text="Общая информация", padding=10)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.info_text = tk.Text(info_frame, height=8, width=60, wrap=tk.WORD, font=('Arial', 10))
        self.info_text.pack(fill=tk.X, padx=5, pady=5)
        self.info_text.insert(tk.END, "Загрузите изображение для начала работы...")
        self.info_text.config(state=tk.DISABLED)
        
        # Статистика
        stats_frame = ttk.LabelFrame(self.info_tab, text="Статистика качества", padding=10)
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X, pady=5)
        
        ttk.Label(stats_grid, text="Емкость:", font=('Arial', 10)).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.capacity_value = ttk.Label(stats_grid, text="-- байт", font=('Arial', 10, 'bold'))
        self.capacity_value.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(stats_grid, text="PSNR:", font=('Arial', 10)).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.psnr_value = ttk.Label(stats_grid, text="-- dB", font=('Arial', 10, 'bold'))
        self.psnr_value.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(stats_grid, text="Размер изображения:", font=('Arial', 10)).grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.size_value = ttk.Label(stats_grid, text="--", font=('Arial', 10))
        self.size_value.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(stats_grid, text="Блоков 8x8:", font=('Arial', 10)).grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.blocks_value = ttk.Label(stats_grid, text="--", font=('Arial', 10))
        self.blocks_value.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Кнопка сохранения
        save_frame = ttk.Frame(self.info_tab)
        save_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.save_btn = ttk.Button(save_frame, text="Сохранить текущее стего-изображение", 
                                   command=self.save_image, state='disabled')
        self.save_btn.pack(side=tk.LEFT, padx=5)
    
    def setup_control_panel(self):
        """Нижняя панель с сообщением и обработкой"""
        control_panel = ttk.Frame(self.root)
        control_panel.pack(fill=tk.X, padx=5, pady=5)
        
        # Поле для сообщения
        ttk.Label(control_panel, text="Сообщение:").pack(side=tk.LEFT, padx=5)
        
        self.message_entry = ttk.Entry(control_panel, width=40)
        self.message_entry.insert(0, "Secret Message")
        self.message_entry.pack(side=tk.LEFT, padx=5)
        
        # Кнопка обработки
        self.process_btn = ttk.Button(control_panel, text="Обработать изображение", 
                                      command=self.process_image, state='disabled')
        self.process_btn.pack(side=tk.LEFT, padx=20)
    
    def load_image(self):
        """Загрузка изображения"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        
        if file_path:
            self.image_path = file_path
            self.file_label.config(text=f"Файл: {file_path.split('/')[-1]}")
            
            # Загружаем и отображаем оригинал
            self.original_image = Image.open(file_path).convert('RGB')
            self.original_viewer.set_image(self.original_image)
            
            # Обновляем информацию
            self.update_info_text("Изображение загружено. Введите сообщение и нажмите 'Обработать'.")
            
            # Активируем кнопку обработки
            self.process_btn.config(state='normal')
    
    def process_image(self):
        """Обработка изображения - создание стегосистемы"""
        if not hasattr(self, 'image_path'):
            messagebox.showwarning("Предупреждение", "Сначала выберите изображение!")
            return
        
        message = self.message_entry.get()
        if not message:
            messagebox.showwarning("Предупреждение", "Введите сообщение!")
            return
        
        try:
            self.root.config(cursor="watch")
            self.root.update()
            
            # Создаем систему стеганографии
            self.stego_system = AdaptiveSteganography(self.image_path, message)
            
            # Активируем элементы управления
            self.level_slider.config(state='normal')
            self.save_btn.config(state='normal')
            
            # Обновляем статистику
            self.update_statistics()
            
            # Отображаем начальный уровень
            self.on_slider_changed(0)
            
            # Переключаемся на вкладку со стего-изображением
            self.notebook.select(1)
            
            self.update_info_text(
                f"Изображение успешно обработано!\n"
                f"Размер: {self.stego_system.width}x{self.stego_system.height}\n"
                f"Блоков 8x8: {self.stego_system.blocks_x * self.stego_system.blocks_y}\n"
                f"Максимальная емкость: {self.stego_system.max_capacity} байт\n\n"
                f"Алгоритм:\n"
                f"• Блоки сортируются по энтропии (от высокой к низкой)\n"
                f"• Встраивание начинается с высокоэнтропийных блоков\n"
                f"• На низких уровнях используются только сложные текстуры\n"
                f"• На высоких уровнях используются даже гладкие области"
            )
            
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
        finally:
            self.root.config(cursor="")
    
    def on_slider_changed(self, value):
        """Обработка изменения ползунка"""
        if not self.stego_system:
            return
        
        level = int(float(value))
        self.level_var.set(level)
        self.current_level = level
        
        # Определяем параметры для текущего уровня
        level_percent = level / 24
        
        # Каналы
        if level < 8:
            channels_text = "Синий"
            channel_color = "blue"
            stage_text = "Этап 1: только высокоэнтропийные блоки"
        elif level < 16:
            channels_text = "Синий+Красный"
            channel_color = "purple"
            stage_text = "Этап 2: среднеэнтропийные блоки"
        else:
            channels_text = "Все каналы"
            channel_color = "black"
            stage_text = "Этап 3: низкоэнтропийные блоки"
        
        # Количество бит
        if level_percent < 0.125:
            bits = 1
        elif level_percent < 0.25:
            bits = 2
        elif level_percent < 0.375:
            bits = 3
        elif level_percent < 0.5:
            bits = 4
        elif level_percent < 0.625:
            bits = 5
        elif level_percent < 0.75:
            bits = 6
        elif level_percent < 0.875:
            bits = 7
        else:
            bits = 8
        
        # Процент используемых блоков
        blocks_percent = int(level / 24 * 100)
        
        # Обновляем метки
        self.level_label.config(
            text=f"Уровень: {level}/24 | {stage_text}",
            foreground=channel_color
        )
        
        self.detail_label.config(
            text=f"Каналы: {channels_text} | Бит на канал: {bits} | Использовано блоков: {blocks_percent}%"
        )
        
        # Получаем и отображаем стего-изображение
        stego_image = self.stego_system.get_image_for_level(level)
        self.stego_viewer.set_image(stego_image)
        
        # Обновляем PSNR
        psnr = self.stego_system.calculate_psnr(self.original_image, stego_image)
        self.psnr_value.config(text=f"{psnr:.2f} dB")
        
        # Обновляем информацию о качестве
        if psnr > 40:
            quality = "Отличное качество"
            quality_color = "green"
        elif psnr > 30:
            quality = "Хорошее качество"
            quality_color = "blue"
        elif psnr > 20:
            quality = "Среднее качество"
            quality_color = "orange"
        else:
            quality = "Плохое качество"
            quality_color = "red"
        
        # Обновляем информационный текст
        self.update_info_text(
            f"Текущий уровень: {level}/24\n"
            f"PSNR: {psnr:.2f} dB - {quality}\n"
            f"Используемые каналы: {channels_text}\n"
            f"Бит на канал: {bits}\n"
            f"Блоков с изменениями: {blocks_percent}%\n"
            f"Стадия: {stage_text}"
        )
    
    def update_statistics(self):
        """Обновление статистики на вкладке информации"""
        if self.stego_system:
            self.capacity_value.config(text=f"{self.stego_system.max_capacity} байт")
            self.size_value.config(text=f"{self.stego_system.width}x{self.stego_system.height}")
            self.blocks_value.config(text=f"{self.stego_system.blocks_x * self.stego_system.blocks_y}")
    
    def update_info_text(self, text):
        """Обновление текстовой информации"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, text)
        self.info_text.config(state=tk.DISABLED)
    
    def save_image(self):
        """Сохранение текущего стего-изображения"""
        if not self.stego_system:
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            stego_image = self.stego_system.get_image_for_level(self.current_level)
            stego_image.save(file_path)
            messagebox.showinfo("Успех", f"Изображение сохранено:\n{file_path}")


def main():
    root = tk.Tk()
    app = SteganographyGUI(root)
    
    # Центрируем окно
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()


if __name__ == '__main__':
    main()