import numpy as np
from PIL import Image
import os
import random

class SteganographyRandomBlock:
    def __init__(self, block_size=32, num_blocks=5):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.end_marker = [0, 0, 0, 0, 0, 0, 0, 0]  # 8 нулевых битов как маркер конца
        
        # Расчет вместимости для одного блока
        self.pixels_per_block = block_size * block_size  # 1024
        self.total_channels = 3  # RGB
        
    def calculate_entropy(self, block):
        """Вычисление энтропии Шеннона для блока"""
        flat_block = block.flatten()
        hist = np.bincount(flat_block, minlength=256)
        hist = hist[hist > 0] / len(flat_block)
        entropy = -np.sum(hist * np.log2(hist))
        return entropy
    
    def find_high_entropy_blocks(self, image_array):
        """Поиск 5 блоков с максимальной энтропией"""
        h, w = image_array.shape[:2]
        block_entropies = []
        
        # Перебираем все возможные блоки 32x32
        for y in range(0, h - self.block_size + 1, self.block_size):
            for x in range(0, w - self.block_size + 1, self.block_size):
                block = image_array[y:y+self.block_size, x:x+self.block_size]
                
                if len(image_array.shape) == 3:
                    entropies = [self.calculate_entropy(block[:,:,c]) for c in range(3)]
                    entropy = np.mean(entropies)
                else:
                    entropy = self.calculate_entropy(block)
                
                block_entropies.append(((x, y), entropy))
        
        # Сортируем по убыванию энтропии и берем первые 5 блоков
        block_entropies.sort(key=lambda x: x[1], reverse=True)
        selected_blocks = [pos for pos, _ in block_entropies[:self.num_blocks]]
        
        print(f"Найдено 5 блоков 32x32 с максимальной энтропией:")
        for i, (x, y) in enumerate(selected_blocks, 1):
            print(f"  Блок {i}: ({x}, {y}), энтропия = {block_entropies[i-1][1]:.4f}")
        
        return selected_blocks
    
    def create_masks(self, image_shape, all_blocks, real_block):
        """
        Создание двух масок:
        - mask_all: все 5 блоков белые
        - mask_real: только реальный блок с информацией белый
        """
        mask_all = np.zeros(image_shape[:2], dtype=np.uint8)
        mask_real = np.zeros(image_shape[:2], dtype=np.uint8)
        
        # Маска со всеми блоками
        for x, y in all_blocks:
            mask_all[y:y+self.block_size, x:x+self.block_size] = 255
        
        # Маска только с реальным блоком
        x, y = real_block
        mask_real[y:y+self.block_size, x:x+self.block_size] = 255
        
        return mask_all, mask_real
    
    def text_to_bits(self, text):
        """Преобразование текста в биты"""
        bits = []
        for char in text:
            char_bits = format(ord(char), '08b')
            bits.extend([int(b) for b in char_bits])
        return bits
    
    def bits_to_text(self, bits):
        """Преобразование битов обратно в текст"""
        chars = []
        for i in range(0, len(bits), 8):
            if i + 8 <= len(bits):
                byte = bits[i:i+8]
                char_code = int(''.join(map(str, byte)), 2)
                if 32 <= char_code <= 126:
                    chars.append(chr(char_code))
        return ''.join(chars)
    
    def calculate_block_capacity(self, n_bits):
        """Расчет вместимости одного блока для заданного количества бит"""
        total_bits = self.pixels_per_block * self.total_channels * n_bits
        return {
            'total_bits': total_bits,
            'total_bytes': total_bits // 8,
            'total_chars': (total_bits // 8) - 1  # минус маркер конца
        }
    
    def generate_random_bits(self, count):
        """Генерация случайных битов"""
        return [random.randint(0, 1) for _ in range(count)]
    
    def embed_in_single_block(self, image_array, message, block_coords, n_bits, fill_entire_block=True):
        """
        Встраивание сообщения в ОДИН конкретный блок
        """
        img_copy = image_array.copy()
        x, y = block_coords
        
        # Получаем вместимость блока
        capacity = self.calculate_block_capacity(n_bits)
        total_bits_in_block = capacity['total_bits']
        
        # Преобразуем сообщение в биты
        message_bits = self.text_to_bits(message)
        message_with_marker = message_bits + self.end_marker
        message_len = len(message_with_marker)
        
        print(f"\n  Встраивание в блок ({x}, {y}):")
        print(f"    Размер блока: {self.block_size}x{self.block_size} = {self.pixels_per_block} пикселей")
        print(f"    Каналов на пиксель: {self.total_channels}")
        print(f"    Вместимость блока при {n_bits} бит/пиксель: {total_bits_in_block} бит")
        print(f"    Битов в сообщении с маркером: {message_len}")
        
        if fill_entire_block:
            # ПОЛНОЕ заполнение блока
            if message_len < total_bits_in_block:
                random_bits_needed = total_bits_in_block - message_len
                random_bits = self.generate_random_bits(random_bits_needed)
                all_bits = message_with_marker + random_bits
                print(f"    Добавлено случайных бит: {random_bits_needed}")
            else:
                all_bits = message_with_marker[:total_bits_in_block]
                print(f"    ВНИМАНИЕ: Сообщение обрезано до {total_bits_in_block} бит")
        else:
            # Встраиваем только сообщение (без заполнения)
            if message_len > total_bits_in_block:
                raise ValueError(f"Сообщение слишком длинное для блока!")
            all_bits = message_with_marker
            print(f"    Встраивание без заполнения: {message_len} бит")
        
        print(f"    ВСЕГО БИТ ДЛЯ ВСТРАИВАНИЯ: {len(all_bits)}")
        
        # Встраиваем биты в блок
        bit_index = 0
        pixels_modified = 0
        changes_log = []
        
        for by in range(y, y + self.block_size):
            for bx in range(x, x + self.block_size):
                if bit_index >= len(all_bits):
                    break
                
                # Для каждого пикселя изменяем n бит в каждом канале
                for c in range(3):
                    if bit_index >= len(all_bits):
                        break
                    
                    current_value = img_copy[by, bx, c]
                    
                    # Берем следующие n бит
                    bits_to_embed = 0
                    for b in range(n_bits):
                        if bit_index < len(all_bits):
                            if all_bits[bit_index] == 1:
                                bits_to_embed |= (1 << b)
                            bit_index += 1
                    
                    # Очищаем n младших бит и вставляем новые
                    mask = (1 << n_bits) - 1
                    new_value = (current_value & ~mask) | bits_to_embed
                    
                    # Логируем первые несколько изменений
                    if len(changes_log) < 5:
                        old_bits = current_value & mask
                        changes_log.append(f"        ({bx},{by},{c}): {current_value:3d} -> {new_value:3d} (биты: {bin(old_bits)[2:].zfill(n_bits)} -> {bin(bits_to_embed)[2:].zfill(n_bits)})")
                    
                    img_copy[by, bx, c] = new_value
                
                pixels_modified += 1
                
            if bit_index >= len(all_bits):
                break
        
        # Выводим примеры изменений
        if changes_log:
            print(f"    Примеры изменений (первые 5):")
            for log in changes_log:
                print(log)
        
        print(f"    Изменено пикселей в блоке: {pixels_modified}")
        print(f"    Использовано бит: {bit_index}")
        
        return img_copy
    
    def extract_from_single_block(self, image_array, block_coords, n_bits, extract_full_block=True):
        """
        Извлечение из ОДНОГО конкретного блока
        """
        x, y = block_coords
        
        if extract_full_block:
            capacity = self.calculate_block_capacity(n_bits)
            total_bits_to_extract = capacity['total_bits']
            print(f"    Извлечение ВСЕХ битов из блока: {total_bits_to_extract} бит")
        else:
            total_bits_to_extract = self.pixels_per_block * self.total_channels * n_bits
            print(f"    Извлечение до маркера конца")
        
        extracted_bits = []
        bits_extracted = 0
        
        for by in range(y, y + self.block_size):
            for bx in range(x, x + self.block_size):
                for c in range(3):
                    if extract_full_block and bits_extracted >= total_bits_to_extract:
                        break
                    
                    pixel_value = image_array[by, bx, c]
                    mask = (1 << n_bits) - 1
                    pixel_bits = pixel_value & mask
                    
                    for b in range(n_bits):
                        if extract_full_block and bits_extracted >= total_bits_to_extract:
                            break
                        bit = (pixel_bits >> b) & 1
                        extracted_bits.append(bit)
                        bits_extracted += 1
                
                if extract_full_block and bits_extracted >= total_bits_to_extract:
                    break
            if extract_full_block and bits_extracted >= total_bits_to_extract:
                break
        
        if not extract_full_block:
            # Ищем маркер конца
            for i in range(len(extracted_bits) - 7):
                if extracted_bits[i:i+8] == self.end_marker:
                    return extracted_bits[:i], extracted_bits
        
        return extracted_bits, extracted_bits
    
    def process_random_block(self, image_path, message, output_dir="stego_random", 
                            fill_entire_block=True, seed=None):
        """
        Выбирает случайный блок из 5 и встраивает в него сообщение
        Сохраняет ДВЕ маски:
        - mask_all.jpg: все 5 блоков белые
        - mask_real.jpg: только реальный блок с информацией
        """
        if seed is not None:
            random.seed(seed)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Загружаем изображение
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        print("=" * 90)
        print("СТЕГАНОГРАФИЯ: СЛУЧАЙНЫЙ БЛОК ИЗ 5 (ДВЕ МАСКИ)")
        print("=" * 90)
        print(f"Изображение: {image_path}")
        print(f"Размер: {img_array.shape[1]}x{img_array.shape[0]}")
        print(f"Сообщение: '{message}'")
        print(f"Длина сообщения: {len(message)} символов = {len(message) * 8} бит")
        print("-" * 90)
        
        # Находим 5 блоков с максимальной энтропией
        all_blocks = self.find_high_entropy_blocks(img_array)
        
        # Выбираем СЛУЧАЙНЫЙ блок из 5
        selected_block_idx = random.randint(0, len(all_blocks) - 1)
        selected_block = all_blocks[selected_block_idx]
        
        print(f"\n🔸 ВЫБРАН СЛУЧАЙНЫЙ БЛОК №{selected_block_idx + 1}: {selected_block}")
        print(f"🔸 Остальные блоки будут в маске all, но без изменений")
        
        # Создаем ДВЕ маски
        mask_all, mask_real = self.create_masks(img_array.shape, all_blocks, selected_block)
        
        # Сохраняем маски
        mask_all_path = os.path.join(output_dir, "mask_all.jpg")
        mask_real_path = os.path.join(output_dir, "mask_real.jpg")
        
        Image.fromarray(mask_all).save(mask_all_path)
        Image.fromarray(mask_real).save(mask_real_path)
        
        print(f"\n✅ Маска со ВСЕМИ блоками: {mask_all_path}")
        print(f"✅ Маска с РЕАЛЬНЫМ блоком: {mask_real_path}")
        
        # Сохраняем оригинал
        original_path = os.path.join(output_dir, "original.jpg")
        img.save(original_path)
        
        # Сохраняем информацию о выбранном блоке в текстовый файл
        info_path = os.path.join(output_dir, "selected_block_info.txt")
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f"Selected block index: {selected_block_idx + 1}\n")
            f.write(f"Selected block coordinates: {selected_block}\n")
            f.write(f"All blocks: {all_blocks}\n")
        
        results = {}
        
        # Для каждого уровня битности (1-8)
        for n_bits in range(1, 9):
            print(f"\n{'='*70}")
            print(f"УРОВЕНЬ {n_bits} БИТ НА ПИКСЕЛЬ")
            print(f"{'='*70}")
            
            try:
                capacity = self.calculate_block_capacity(n_bits)
                print(f"Вместимость одного блока: {capacity['total_chars']} символов, {capacity['total_bits']} бит")
                
                # Встраиваем ТОЛЬКО в выбранный случайный блок
                embedded_array = self.embed_in_single_block(
                    img_array.copy(), 
                    message, 
                    selected_block, 
                    n_bits,
                    fill_entire_block=fill_entire_block
                )
                
                # Сохраняем изображение
                fill_type = "full" if fill_entire_block else "message_only"
                output_filename = f"embedded_{n_bits}bits_{fill_type}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                
                embedded_img = Image.fromarray(embedded_array)
                embedded_img.save(output_path)
                
                # Извлекаем из выбранного блока для проверки
                extracted_message_bits, all_extracted = self.extract_from_single_block(
                    embedded_array, 
                    selected_block, 
                    n_bits,
                    extract_full_block=fill_entire_block
                )
                extracted_message = self.bits_to_text(extracted_message_bits)
                
                # Проверяем изменения в разных блоках
                changes_in_selected = 0
                changes_in_others = 0
                
                for i, (bx, by) in enumerate(all_blocks):
                    block_changes = 0
                    for y in range(by, by + self.block_size):
                        for x in range(bx, bx + self.block_size):
                            if not np.array_equal(img_array[y, x], embedded_array[y, x]):
                                block_changes += 1
                    
                    if i == selected_block_idx:
                        changes_in_selected = block_changes
                    else:
                        changes_in_others += block_changes
                
                results[n_bits] = {
                    'path': output_path,
                    'selected_block': selected_block_idx + 1,
                    'extracted': extracted_message,
                    'correct': extracted_message == message,
                    'changes_in_selected': changes_in_selected,
                    'changes_in_others': changes_in_others,
                    'capacity': capacity
                }
                
                print(f"\n  РЕЗУЛЬТАТ:")
                print(f"  Сохранено: {output_path}")
                print(f"  Изменено в ВЫБРАННОМ блоке {selected_block_idx + 1}: {changes_in_selected} пикселей")
                print(f"  Изменено в ДРУГИХ блоках: {changes_in_others} пикселей")
                print(f"  Извлечено: '{extracted_message}'")
                print(f"  Корректно: {extracted_message == message}")
                
            except Exception as e:
                print(f"  ОШИБКА: {e}")
                results[n_bits] = {'error': str(e)}
        
        # Сохраняем отчет
        self.save_report(results, message, all_blocks, selected_block_idx, output_dir)
        
        return results, all_blocks, selected_block_idx, (mask_all, mask_real)
    
    def save_report(self, results, original_message, all_blocks, selected_idx, output_dir):
        """Сохранение отчета с информацией о двух масках"""
        report_path = os.path.join(output_dir, "report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 90 + "\n")
            f.write("ОТЧЕТ: СЛУЧАЙНЫЙ БЛОК ИЗ 5 (ДВЕ МАСКИ)\n")
            f.write("=" * 90 + "\n\n")
            
            f.write(f"Исходное сообщение: '{original_message}'\n")
            f.write(f"Длина: {len(original_message)} символов = {len(original_message) * 8} бит\n\n")
            
            f.write("Все 5 блоков с максимальной энтропией:\n")
            for i, (x, y) in enumerate(all_blocks, 1):
                marker = "🔸 РЕАЛЬНЫЙ" if i-1 == selected_idx else "  "
                f.write(f"  {marker} Блок {i}: ({x}, {y})\n")
            
            f.write(f"\nСохраненные маски:\n")
            f.write(f"  - mask_all.jpg: все 5 блоков белые (для запутывания)\n")
            f.write(f"  - mask_real.jpg: только блок {selected_idx + 1} белый (реальный)\n\n")
            
            f.write("РЕЗУЛЬТАТЫ ПО УРОВНЯМ:\n")
            f.write("-" * 90 + "\n")
            
            for n_bits in range(1, 9):
                f.write(f"\n[{n_bits} БИТ НА ПИКСЕЛЬ]\n")
                
                if n_bits in results:
                    if 'error' in results[n_bits]:
                        f.write(f"  Статус: ОШИБКА - {results[n_bits]['error']}\n")
                    else:
                        cap = results[n_bits]['capacity']
                        f.write(f"  Файл: {os.path.basename(results[n_bits]['path'])}\n")
                        f.write(f"  Вместимость блока: {cap['total_chars']} символов, {cap['total_bits']} бит\n")
                        f.write(f"  Изменено в реальном блоке: {results[n_bits]['changes_in_selected']} пикселей\n")
                        f.write(f"  Изменено в других блоках: {results[n_bits]['changes_in_others']} пикселей\n")
                        f.write(f"  Извлечено: '{results[n_bits]['extracted']}'\n")
                        f.write(f"  Корректно: {results[n_bits]['correct']}\n")
                else:
                    f.write(f"  Нет данных\n")
        
        print(f"\n📄 Отчет сохранен: {report_path}")

def visualize_masks(image_shape, all_blocks, real_block, output_dir):
    """Визуализация двух масок для наглядности"""
    
    # Создаем цветную визуализацию
    vis = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    
    # Рисуем все блоки серым
    for x, y in all_blocks:
        vis[y:y+32, x:x+32] = [100, 100, 100]
    
    # Рисуем реальный блок красным
    x, y = real_block
    vis[y:y+32, x:x+32] = [255, 0, 0]
    
    # Добавляем границы блоков
    for x, y in all_blocks:
        # Горизонтальные линии
        vis[y:y+32, x:x+2] = [255, 255, 255]
        vis[y:y+2, x:x+32] = [255, 255, 255]
        vis[y+30:y+32, x:x+32] = [255, 255, 255]
        vis[y:y+32, x+30:x+32] = [255, 255, 255]
    
    vis_path = os.path.join(output_dir, "blocks_visualization.jpg")
    Image.fromarray(vis).save(vis_path)
    print(f"🎨 Визуализация блоков: {vis_path}")
    print(f"   - Серые блоки: все 5 кандидатов")
    print(f"   - Красный блок: реальный с информацией")

def demonstrate_random_block():
    """Демонстрация случайного выбора блока с двумя масками"""
    
    # Создаем экземпляр класса
    stego = SteganographyRandomBlock(block_size=32, num_blocks=5)
    
    # Тестовое сообщение
    message = "Secret Message"
    
    # Путь к изображению
    input_image = "container.jpg"  # Замените на ваш файл
    
    try:
        # Запускаем обработку
        results, all_blocks, selected_idx, masks = stego.process_random_block(
            image_path=input_image,
            message=message,
            output_dir="stego_dual_masks",
            fill_entire_block=True,
            seed=42  # для воспроизводимости
        )
        
        mask_all, mask_real = masks
        
        # Создаем визуализацию
        img = Image.open(input_image).convert('RGB')
        visualize_masks(np.array(img).shape, all_blocks, all_blocks[selected_idx], "stego_dual_masks")
        
        print("\n" + "=" * 90)
        print("ИТОГОВАЯ ИНФОРМАЦИЯ:")
        print("=" * 90)
        print(f"✅ Всего создано файлов в директории 'stego_dual_masks':")
        print(f"   - mask_all.jpg: все 5 блоков белые")
        print(f"   - mask_real.jpg: только блок {selected_idx + 1} белый")
        print(f"   - blocks_visualization.jpg: серый - все блоки, красный - реальный")
        print(f"   - embedded_1bits_full.jpg ... embedded_8bits_full.jpg: изображения с информацией")
        print(f"   - report.txt: подробный отчет")
        print(f"   - selected_block_info.txt: информация о выбранном блоке")
        
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    demonstrate_random_block()