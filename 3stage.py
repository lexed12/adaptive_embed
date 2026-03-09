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
        
        # Предвычисление энтропии блоков
        self.block_entropies = self._calculate_block_entropies()
        
        # Сортировка блоков по энтропии (от высокой к низкой)
        self.sorted_blocks = self._sort_blocks_by_entropy()
        
        # Максимальная емкость
        self.max_capacity = self._calculate_max_capacity()
        
        # Генерация изображений для разных уровней
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
        bits_per_block = self.block_size * self.block_size * 3 * 2
        max_bits = total_blocks * bits_per_block
        return max_bits // 8
    
    def _embed_message(self, level):
        stego_array = self.image_array.copy()
        
        num_blocks = int(len(self.sorted_blocks) * level)
        
        if level < 0.2:
            bits_per_channel = 1
        elif level < 0.4:
            bits_per_channel = 2
        elif level < 0.6:
            bits_per_channel = 3
        else:
            bits_per_channel = 4
        
        message_bytes = self.message.encode('utf-8')
        message_bits = ''.join(format(byte, '08b') for byte in message_bytes)
        
        if len(message_bits) < num_blocks * self.block_size * self.block_size * 3 * bits_per_channel:
            repeats = (num_blocks * self.block_size * self.block_size * 3 * bits_per_channel) // len(message_bits) + 1
            message_bits = (message_bits * repeats)[:num_blocks * self.block_size * self.block_size * 3 * bits_per_channel]
        
        bit_index = 0
        
        for block_info in self.sorted_blocks[:num_blocks]:
            y = block_info['y']
            x = block_info['x']
            
            y_start = y * self.block_size
            y_end = (y + 1) * self.block_size
            x_start = x * self.block_size
            x_end = (x + 1) * self.block_size
            
            for i in range(self.block_size):
                for j in range(self.block_size):
                    for channel in range(3):
                        if bit_index < len(message_bits):
                            pixel_value = stego_array[y_start + i, x_start + j, channel]
                            
                            if bits_per_channel == 1:
                                bit = int(message_bits[bit_index])
                                stego_array[y_start + i, x_start + j, channel] = (pixel_value & 0xFE) | bit
                                bit_index += 1
                            else:
                                bits_to_embed = message_bits[bit_index:bit_index + bits_per_channel]
                                if len(bits_to_embed) == bits_per_channel:
                                    value_to_embed = int(bits_to_embed, 2)
                                    mask = (0xFF << bits_per_channel) & 0xFF
                                    stego_array[y_start + i, x_start + j, channel] = (pixel_value & mask) | value_to_embed
                                    bit_index += bits_per_channel
        
        return Image.fromarray(stego_array.astype('uint8'))
    
    def _generate_stego_levels(self):
        levels = 10
        stego_images = []
        
        for i in range(levels + 1):
            level = i / levels
            stego_image = self._embed_message(level)
            stego_images.append(stego_image)
        
        return stego_images
    
    def get_image_for_level(self, level_index):
        return self.stego_images[level_index]
    
    def calculate_psnr(self, original, stego):
        """
        Вычисление PSNR для оригинальных (немасштабированных) изображений
        """
        # Убеждаемся, что оба изображения имеют одинаковый размер
        if original.size != stego.size:
            # Если размеры разные, изменяем размер stego до размера original
            stego = stego.resize(original.size, Image.Resampling.LANCZOS)
        
        original_array = np.array(original)
        stego_array = np.array(stego)
        
        mse = np.mean((original_array - stego_array) ** 2)
        if mse == 0:
            return float('inf')
        
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        return psnr


class SteganographyTkinterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Адаптивная стеганография на основе энтропии")
        self.root.geometry("1200x800")
        
        self.stego_system = None
        self.current_level = 0
        self.original_image_display = None  # Для отображения
        self.original_image_full = None     # Полноразмерное оригинальное изображение
        
        self.setup_ui()
    
    def setup_ui(self):
        # Основной фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Настройка растягивания
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Верхняя панель
        control_frame = ttk.LabelFrame(main_frame, text="Управление", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        control_frame.columnconfigure(1, weight=1)
        
        ttk.Button(control_frame, text="Выбрать изображение", 
                  command=self.load_image).grid(row=0, column=0, padx=5)
        
        self.image_label = ttk.Label(control_frame, text="Изображение не выбрано")
        self.image_label.grid(row=0, column=1, padx=5, sticky=tk.W)
        
        ttk.Label(control_frame, text="Сообщение:").grid(row=1, column=0, padx=5, pady=5)
        self.message_entry = ttk.Entry(control_frame, width=50)
        self.message_entry.insert(0, "Secret Message")
        self.message_entry.grid(row=1, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Button(control_frame, text="Обработать", 
                  command=self.process_image).grid(row=1, column=2, padx=5)
        
        # Панель с изображениями
        images_frame = ttk.Frame(main_frame)
        images_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        images_frame.columnconfigure(0, weight=1)
        images_frame.columnconfigure(1, weight=1)
        images_frame.rowconfigure(0, weight=1)
        
        # Оригинал
        original_frame = ttk.LabelFrame(images_frame, text="Оригинал")
        original_frame.grid(row=0, column=0, padx=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        original_frame.columnconfigure(0, weight=1)
        original_frame.rowconfigure(0, weight=1)
        
        self.original_canvas = tk.Canvas(original_frame, width=500, height=400, bg='#f0f0f0', highlightthickness=0)
        self.original_canvas.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Стего
        stego_frame = ttk.LabelFrame(images_frame, text="Стего-изображение")
        stego_frame.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        stego_frame.columnconfigure(0, weight=1)
        stego_frame.rowconfigure(0, weight=1)
        
        self.stego_canvas = tk.Canvas(stego_frame, width=500, height=400, bg='#f0f0f0', highlightthickness=0)
        self.stego_canvas.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Ползунок
        slider_frame = ttk.Frame(stego_frame)
        slider_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        slider_frame.columnconfigure(1, weight=1)
        
        ttk.Label(slider_frame, text="Минимум").grid(row=0, column=0, padx=5)
        
        self.level_var = tk.DoubleVar(value=0)
        self.level_slider = ttk.Scale(
            slider_frame, 
            from_=0, to=10, 
            orient=tk.HORIZONTAL,
            variable=self.level_var,
            command=self.on_slider_changed,
            state='disabled'
        )
        self.level_slider.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=10)
        
        ttk.Label(slider_frame, text="Максимум").grid(row=0, column=2, padx=5)
        
        self.level_display = ttk.Label(stego_frame, text="Уровень: 0/10")
        self.level_display.grid(row=2, column=0, pady=5)
        
        # Нижняя панель
        stats_frame = ttk.LabelFrame(main_frame, text="Статистика", padding="10")
        stats_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.capacity_label = ttk.Label(stats_frame, text="Емкость: -- байт")
        self.capacity_label.pack(side=tk.LEFT, padx=10)
        
        self.psnr_label = ttk.Label(stats_frame, text="PSNR: -- dB")
        self.psnr_label.pack(side=tk.LEFT, padx=10)
        
        self.warning_label = ttk.Label(stats_frame, text="", foreground='red')
        self.warning_label.pack(side=tk.LEFT, padx=10)
        
        self.save_button = ttk.Button(stats_frame, text="Сохранить", 
                                     command=self.save_image, state='disabled')
        self.save_button.pack(side=tk.RIGHT, padx=5)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        
        if file_path:
            self.image_path = file_path
            self.image_label.config(text=f"Файл: {file_path.split('/')[-1]}")
            
            # Загружаем и сохраняем полноразмерное изображение
            self.original_image_full = Image.open(file_path).convert('RGB')
            
            # Создаем уменьшенную копию для отображения
            display_image = self.original_image_full.copy()
            display_image.thumbnail((500, 400), Image.Resampling.LANCZOS)
            self.original_image_display = ImageTk.PhotoImage(display_image)
            
            # Отображаем на canvas
            self.original_canvas.delete("all")
            self.original_canvas.create_image(
                self.original_canvas.winfo_width() // 2,
                self.original_canvas.winfo_height() // 2,
                image=self.original_image_display,
                anchor=tk.CENTER
            )
    
    def process_image(self):
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
            self.save_button.config(state='normal')
            
            # Обновляем информацию о емкости
            self.capacity_label.config(text=f"Емкость: {self.stego_system.max_capacity} байт")
            
            # Отображаем начальный уровень
            self.on_slider_changed(0)
            
            messagebox.showinfo("Успех", 
                f"Изображение успешно обработано!\n"
                f"Размер блока: 8x8\n"
                f"Всего блоков: {self.stego_system.blocks_y * self.stego_system.blocks_x}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
        finally:
            self.root.config(cursor="")
    
    def on_slider_changed(self, value):
        if not self.stego_system:
            return
        
        # Округляем до целого для дискретных значений
        level = int(float(value))
        self.level_var.set(level)  # Фиксируем на целых значениях
        self.current_level = level
        self.level_display.config(text=f"Уровень: {level}/10")
        
        # Получаем полноразмерное стего-изображение
        stego_image_full = self.stego_system.get_image_for_level(level)
        
        # Создаем уменьшенную копию для отображения
        display_image = stego_image_full.copy()
        display_image.thumbnail((500, 400), Image.Resampling.LANCZOS)
        self.stego_image_display = ImageTk.PhotoImage(display_image)
        
        # Отображаем на canvas
        self.stego_canvas.delete("all")
        self.stego_canvas.create_image(
            self.stego_canvas.winfo_width() // 2,
            self.stego_canvas.winfo_height() // 2,
            image=self.stego_image_display,
            anchor=tk.CENTER
        )
        
        # Вычисляем PSNR на полноразмерных изображениях
        psnr = self.stego_system.calculate_psnr(
            self.original_image_full,
            stego_image_full
        )
        self.psnr_label.config(text=f"PSNR: {psnr:.2f} dB")
        
        # Показываем предупреждение при высоком уровне
        if level >= 8:
            self.warning_label.config(text="⚠️ Внимание: заметные артефакты!")
        else:
            self.warning_label.config(text="")
    
    def save_image(self):
        if not self.stego_system:
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            # Сохраняем полноразмерное изображение
            stego_image = self.stego_system.get_image_for_level(self.current_level)
            stego_image.save(file_path)
            messagebox.showinfo("Успех", f"Изображение сохранено:\n{file_path}")


def main():
    root = tk.Tk()
    app = SteganographyTkinterGUI(root)
    
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